# Symbolic IVP Preparation Story Tests

## IVP Jacobian Scratch Reuse (2026-10-23)

### Audit and change

The Expr row-major `try_evaluate_jacobian_row_major_into_with_workspace`
compatibility API reused flattened arguments but created and dropped its
structurally sparse scalar-value `Vec` on every callback. The new
`...with_workspaces` form accepts both caller-owned buffers; the original
signature remains source-compatible. BDF's solver-lifetime Expr Jacobian
workspace previously reserved `n*n` doubles even though its prepared Expr plan
stores only nonzero derivative entries. It now reserves exactly that sparse
value capacity. The dense Jacobian output required by the current BDF solver
ABI is unchanged.

`IvpTelemetrySnapshot::allocated_bytes` is explicitly the sum of instrumented
scratch-buffer capacity growth/preallocation. It is not a global allocator or
RSS measurement. The BDF preparation snapshot is taken after callback scratch
construction so this known workspace allocation is included.

### Debug correctness

`expr_row_major_workspace_reuses_sparse_values_and_matches_owned_jacobian`
compares owned, compatibility and reusable-workspace Jacobians on a 64-state
cyclic nearest-neighbor system. All row-major values match exactly. The system
has 128 structurally nonzero derivatives rather than 4096 dense slots; repeated
caller-owned callbacks leave both capacities and the telemetry allocation
counter unchanged after warmup. The compatibility method still evaluates
correctly and its counter exposes the repeated temporary value-buffer growth.
The shared four-mode telemetry contract and BDF symbolic-preparation snapshot
tests also pass in Debug.

### Release callback A/B

Command:

```text
cargo bench --bench be_symbolic_frontend_benches -- 'be_symbolic_ivp_workspace_reuse'
```

Criterion used 30 samples per case, 3-second warmup and 5-second measurement
time. The callback writes the same dense row-major output in both routes;
preparation is outside the timed callback. Median/MAD in microseconds:

| n | compatibility temporary | caller-owned scratch | change | sparse vs prior BDF scratch |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.206/0.024 | 1.154/0.013 | -4.3% | 3,072 B vs 131,072 B |
| 256 | 2.967/0.053 | 2.910/0.050 | -1.9% | 6,144 B vs 524,288 B |
| 512 | 16.273/0.098 | 16.103/0.135 | -1.0% | 12,288 B vs 2,097,152 B |

The callback timing gain shrinks as dense output zeroing dominates. The
workspace-memory reduction is large and scales with sparsity, but these are
component measurements, not evidence of a full BDF solve speedup. Keep both
symbolic frontends and solver defaults unchanged. Before claiming end-to-end
performance benefit, run BDF's large full-solve gate and compare its telemetry-
Off wall time, trajectory and discrete work against the existing machine
baseline. No allocator-wide count or peak RSS conclusion is made here.

### BDF Full-Solve Diagnostic (2026-10-23)

The existing `bdf_diff256_lambdify_expr_seq` perf gate was run with 101 samples
against the local `windows-dev` baseline. It returned `UNSTABLE`: the first
series measured `18.536 ms` baseline versus `9.668 ms` current (-47.8%), and
the confirmation series measured `9.898 ms` current, but current relative MAD
was about 5.3%, above the configured 5% stability limit. Correctness and all
discrete work counters matched exactly (22 accepted steps, 25 candidates, 3
rejections, 56 RHS calls, 1 Jacobian evaluation and 7 factorizations).

This is not evidence for the scratch-reuse change: the current run used a dirty
tree with other IVP symbolic changes, and backend preparation also shifted from
`13.160 ms` to `3.390 ms`, far beyond the expected scope of reserving the
Jacobian value buffer. The run report is
`performance/runs/1792775427153-22188/summary.md`. A clean, isolated BDF
full-solve A/B remains open; the callback-only measurements above and exact
scratch-capacity reduction are the evidence attributable to this change.

## ExprLegacy sparse Jacobian dependency scan (2026-10-22)

### Hypothesis

Large IVP systems often have locally coupled equations. The ExprLegacy
preparation path previously differentiated every equation against every state
variable, even when most partial derivatives were structurally zero. A single
dependency scan per equation should reduce cold symbolic work for sparse
Jacobians, while small systems may be faster with the simpler full scan.

### Implementation and semantic contract

- For fewer than 64 state columns, retain the original full differentiation
  loop. The crossover came from the sparse-chain A/B measurements below.
- At 64 or more columns, discover each equation's sorted variable set once and
  differentiate only against state variables present in that set.
- Preserve the dense Jacobian shape, caller-provided column order, duplicate
  columns, and explicit `Expr::Const(0.0)` entries for absent dependencies.
- Skip `simplify()` only for already-constant expressions. Non-constant
  derivatives continue through the existing simplifier; AtomView and AOT paths
  are unchanged.

### Correctness gates

- `cargo test --lib symbolic::symbolic_ivp::tests::`: 18 passed.
- `cargo test --lib numerical::LSODE2::tests::lsode2_exprlegacy_and_atomview_native_solver_traces_match_in_debug`:
  passed. ExprLegacy and AtomViewNative solver traces remained in parity.
- The new 64-state unit test uses reversed state-column order, checks structural
  zeros and verifies selected analytic Jacobian entries through the prepared
  callback.

### Release measurements

Command:

```text
cargo bench --bench be_symbolic_frontend_benches -- 'sparse-chain'
```

Criterion used 10 samples per case with the bench's 3-second warmup. The
fixture is a cyclic nearest-neighbor chain; each equation depends on three of
the state variables. Results are diagnostic measurements on one Windows
machine, not portable baselines.

| states | full Jacobian reference | dependency-aware Jacobian | full ExprLegacy preparation |
| ---: | ---: | ---: | ---: |
| 32 | 40.1 us | 60.0 us | 85.9 us |
| 64 | 160.3 us | 147.3 us | 254.6 us |
| 128 | 576.7 us | 305.2 us | 788.9 us |
| 256 | 2.277 ms | 822.6 us | 2.780 ms |

The optimized Jacobian builder is slower at 32 states, slightly faster at 64,
and about 47% and 64% faster at 128 and 256 states, respectively. The
full-preparation Criterion comparison against the preceding implementation
pass reported improvements of about 9%, 11%, and 12% at 64, 128, and 256
states. At 32 states the fallback avoided a regression. The full-preparation
gain is smaller than the isolated Jacobian gain because callback preparation
and compilation remain part of the measured operation.

## Skip structural Jacobian zeros (2026-10-22)

### Hypothesis

Sparse and banded Expr IVP routes were paying twice for structural zeros:
shared dense callbacks lambdified and evaluated every matrix cell, while LSODE2
Sparse/Banded first materialized a dense symbolic matrix and then scanned it
back into nonzero coordinates. A coordinate-form ExprLegacy plan should reduce
cold preparation for large sparse systems. Dense callback APIs must retain a
dense output matrix, but need not compile or evaluate its exact-zero entries.

### Implementation and semantic contract

- ExprLegacy differentiation now creates an ordered coordinate plan and drops
  positive structural zeros. The existing dense Jacobian builder materializes
  the same `rows x cols` matrix from this plan, preserving explicit positive
  zeros, negative-zero sign, and caller column order. LSODE2 sparse layouts
  continue to omit both signed zeros, matching the former dense-scan path.
- LSODE2 SparseTriplets and Banded ExprLegacy preparation consumes the
  coordinate plan directly; it no longer creates an intermediate dense
  symbolic matrix. Dense and AtomView routes retain their existing preparation
  structures.
- The shared dense callback elides positive structural zeros while retaining
  negative-zero expressions for IEEE sign preservation, then assembles the
  established dense `DMatrix` output initialized with zeros.
- No callback ABI, residual, Jacobian values, sparse entry ordering, linear
  solver, frontend selection, or numerical scheme changed.

### Debug correctness gates

- `cargo test --lib symbolic::symbolic_ivp::tests::`: 19 passed.
- `cargo test --lib numerical::LSODE2::native_jacobian::tests::`: 19 passed.
- The 64-state shared ExprLegacy test checks reversed columns, selected
  analytic entries, retained dense shape/zeros, and exactly four scalar
  evaluations for four structural nonzeros.
- A focused IEEE regression verifies that dense callback zero elision retains
  the sign bit of `-0.0`; LSODE2 separately verifies sparse compilation drops
  both signed-zero triplets as before.
- The LSODE2 64-state test compares Dense, SparseTriplets and Banded values;
  Sparse and Banded each evaluate exactly 127 structural entries and retain
  componentwise parity with Dense.

### Release A/B measurements

Command:

```text
cargo bench --bench be_symbolic_frontend_benches -- be_symbolic_ivp_sparse_jacobian_ab --noplot
```

Criterion used 50 samples per case, a 3-second warmup and 5-second measurement
windows. The fixture is a locally coupled chain with two or three nonzeros per
row. The LSODE2 control reproduces the previous dense symbolic build, dense
nonzero scan and nonzero lambdification; the candidate calls the production
SparseTriplets constructor. Numbers are this Windows machine's diagnostics,
not portable performance promises.

| states | LSODE2 cold prepare: dense-scan control | direct sparse plan | speedup |
| ---: | ---: | ---: | ---: |
| 128 | 612.33 us | 287.04 us | 2.13x |
| 256 | 2.3598 ms | 772.46 us | 3.05x |
| 512 | 9.4022 ms | 2.3124 ms | 4.07x |

| states | shared dense callback: all-cell control | zero-skip callback | speedup |
| ---: | ---: | ---: | ---: |
| 128 | 60.613 us | 1.4675 us | 41.3x |
| 256 | 299.19 us | 3.5487 us | 84.3x |
| 512 | 2.0141 ms | 258.02 us | 7.8x |

The shared callback still allocates and zero-initializes the required dense
output, which becomes a significant floor at 512 states. LSODE2's warm
SparseTriplets callback did not get faster: control/candidate medians were
0.531/0.567 us, 1.069/1.242 us and 2.810/2.843 us at 128/256/512. That stage
already evaluated only nonzero entries before this change; the improvement is
in preparation, not a demonstrated warm sparse-evaluation gain. This A/B does
not establish an end-to-end solver speedup and does not include an IVP solve.

### Scope and follow-up

This supports the cold-preparation optimization for sparse, locally coupled
systems and zero-elision in the shared dense callback on sparse patterns. It
does not demonstrate faster LSODE2 warm Sparse callbacks, solver integration,
end-to-end solve time, dense-system performance, or a benefit for all equation
shapes. The differentiation crossover is currently a conservative fixed
column threshold and should be revisited only with additional representative
workloads. No default frontend, linear solver, runtime callback ABI, or Atom
implementation changed.

## Known-band ExprLegacy preparation (2026-10-22)

### Hypothesis and scope

When LSODE2 receives an explicit Banded `(kl, ku)` layout, its ExprLegacy
preparation can restrict candidate state columns to that band instead of
checking all `n` columns in every equation. This is distinct from generic
Sparse preparation: the declared band supplies a bounded column range. For a
valid band, out-of-band state variables are not differentiated or lambdified.
To preserve previous invalid-layout behavior, dependencies actually found
outside the declared band are still differentiated; a nonzero entry remains
in the plan and Banded callback assembly rejects it rather than silently
dropping it.

Preparation discovers equation dependencies once, then differentiates rows
using Rayon only when the estimated relevant work can support at least two
jobs with a conservative minimum of 256 state dependencies per useful worker.
The effective worker count is capped by available rows and estimated useful
work, so high machine core counts do not suppress parallelism for a large but
bounded Banded task. This changes only cold ExprLegacy Jacobian preparation;
numerical callbacks, matrix values, solver scheme, backend choice and callback
execution policy are unchanged.

### Debug correctness gates

- `cargo test --lib symbolic::symbolic_ivp::tests::`: 20 passed.
- `cargo test --lib numerical::LSODE2::native_jacobian::tests::`: 20 passed.
- `cargo test --lib lsode2_debug_wider_boundary_sparse_and_banded_layouts_match_dense`: passed.
- `cargo test --lib lsode2_exprlegacy_and_atomview_native_solver_traces_match_in_debug`: passed.
- A 520-state chain compares the band-restricted plan with the general sparse
  ExprLegacy plan inside a forced four-worker Rayon pool. A second assertion
  verifies that a nonzero out-of-band derivative is retained and rejected by
  callback assembly.
- The LSODE2 callback test compares Dense, SparseTriplets and explicit-width
  Banded values and confirms matching scalar-evaluation counts.

### Release A/B diagnostic

Command:

```text
cargo bench --bench be_symbolic_frontend_benches -- lsode2-banded-jacobian-prepare --noplot
```

Criterion used 50 samples, a 3-second warmup and 5-second measurement per
case. It compared solver-ready cold constructors on a tridiagonal chain. The
parallel variants use prebuilt local Rayon pools; pool creation is outside
the timed closure, while `pool.install` scheduling is included. Times are
medians in milliseconds and are machine-local observations:

| states | generic Sparse / global | known Banded / global Auto | known Banded / 1 worker | known Banded / 4 workers |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.965 | 0.486 | 0.464 | 0.246 |
| 1024 | 25.179 | 1.222 | 1.784 | 0.826 |
| 2048 | 97.837 | 2.249 | 3.525 | 1.588 |

In this run the known-width route was about 4.0x, 20.6x and 43.5x faster than
the Sparse constructor at these sizes. The forced four-worker Banded route
was about 1.9x, 2.2x and 2.2x faster than its one-worker counterpart. An
earlier release pass showed the same large-system direction but materially
different Sparse control times, so the Sparse-vs-Banded ratios are diagnostic,
not yet a portable or end-to-end performance guarantee. The paired
one-worker/four-worker Banded measurements are more consistent, but still
need confirmation in full solver workloads.

### Conclusion

The hypothesis is supported for cold ExprLegacy preparation with an explicit
band: restricting candidate columns avoids the generic all-column scan, and
parallel row preparation helps the tested 1024/2048 workloads. No claim is
made about total LSODE2 solve time, AtomView, AOT, unspecified/inferred
bandwidth, or arbitrary equation structures. Keep the conservative dispatch
and retain both Sparse and Banded routes. A larger LSODE2 workload release
story should confirm the preparation gain in a full solve before any broader
symbolic API or solver-default changes.

## IVP-wide multi-target differentiation and row parallelism (2026-10-22)

### Hypothesis and design

LSODE2 is the pilot, but the ExprLegacy Jacobian builder is shared by BDF,
Radau and BE as well. The experiment separates two ideas that must not be
confounded: computing several derivatives in one recursive traversal, and
parallelizing independent Jacobian rows. AtomView, AOT callbacks, solver
iteration logic and numerical schemes are outside this change.

The release-only A/B test used two warmups and 15 timed samples per route, with
local Rayon pools constructed outside the timed operation and `pool.install`
included. Rows in the sparse-chain fixture reference at most three state
variables. Dense-row fixtures combine nonlinear functions of every target.
All routes are checked for derivative/plan parity before timing.

### Release preparation A/B

Sparse-chain Jacobian preparation, milliseconds median/MAD. `hybrid` is the
production generic ExprLegacy builder; `batch` is an isolated multi-target
reference, and pools are created outside the timed region:

| states | scalar-1 | scalar-4 | batch-1 | batch-4 | hybrid-1 | hybrid-4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.933/0.009 | 0.288/0.003 | 1.029/0.011 | 0.302/0.003 | 0.639/0.017 | 0.260/0.011 |
| 1024 | 8.218/0.025 | 2.340/0.032 | 8.628/0.108 | 2.320/0.038 | 3.217/0.034 | 0.857/0.022 |
| 2048 | 30.069/0.221 | 8.278/0.064 | 30.216/0.122 | 7.935/0.046 | 4.358/0.052 | 1.550/0.051 |

Dense single-row derivative preparation, milliseconds median/MAD:

| targets in row | scalar-per-target | multi-target |
| ---: | ---: | ---: |
| 8 | 0.096/0.020 | 0.036/0.000 |
| 32 | 0.347/0.001 | 0.185/0.001 |

Explicit LSODE2 Banded preparation at `(1, 1)`, milliseconds median/MAD:

| states | scalar valid-band reference, 1 worker | scalar valid-band reference, 4 workers | production checked band path, 1 worker | production checked band path, 4 workers |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.510/0.004 | 0.201/0.008 | 0.665/0.015 | 0.218/0.008 |
| 1024 | 1.998/0.009 | 0.689/0.015 | 2.411/0.017 | 0.822/0.021 |
| 2048 | 4.006/0.032 | 1.260/0.024 | 4.715/0.038 | 1.489/0.018 |

### Decision and limitations

The production generic builder is about 1.46x, 2.55x and 6.90x faster than
scalar-1 at 256/1024/2048 states, and about 1.11x, 2.73x and 5.34x faster than
scalar-4. The gain comes from visiting only actual equation dependencies;
isolated batch differentiation alone does not help sparse rows. For wide
dense rows, multi-target differentiation is about 2.0x faster at both tested
widths. Four-worker row scheduling also helps the generic sparse workload,
with the 2048-state hybrid falling from 4.358 ms to 1.550 ms.

The explicit Banded route deliberately remains scalar within its declared
narrow band. The checked production path is slower than the valid-band scalar
reference by about 18-30%; that reference does not perform the compatibility
scan that retains out-of-band derivatives so invalid user bandwidths fail
instead of silently truncating the Jacobian. The new direct band scan did not
remove that cost, and these data do not justify claiming a Banded preparation
speedup. The safety behavior is retained; any attempt to reduce its cost must
keep the invalid-band regression tests.

The measurements above cover cold Jacobian preparation only. The follow-up
solver-level gate below checks preparation and full-solve behavior for the
optimized ExprLegacy route.

The acceptance policy is large-workload-first: a small absolute overhead is
acceptable if the full solve shows a robust gain on large or very large
systems. Introduce a size-based algorithm dispatch only for a measured,
material small-workload loss; do not optimize away a large-system win to avoid
an unnoticeable few milliseconds.

## Solver-Level ExprLegacy Gate (2026-10-23)

The LSODE2 release gate reused the production diffusion-chain workload and
compared ExprLegacy with AtomViewNative at `n=1024/2048`, for Sparse and
Banded layouts. Each route had 15 repetitions; telemetry was Detailed, the
callback policy was Sequential, and AOT was excluded. The existing story runs
each route in blocks rather than alternating individual pairs, so the table
below is a repeated workload comparison, not a host-order-controlled paired
benchmark. Times are mean +/- standard deviation in milliseconds; stage
timings are representative telemetry samples and nested scopes are not
additive.

| n | layout | route | prepare ms | solve ms | total ms | residual/Jacobian/linear calls | max final-state diff |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| 1024 | Sparse | ExprLegacy | 11.674 +/- 0.522 | 63.030 +/- 0.779 | 87.048 +/- 1.502 | 393 / 234 / 388 | reference |
| 1024 | Sparse | AtomViewNative | 18.703 +/- 0.409 | 65.584 +/- 0.612 | 96.351 +/- 0.829 | 393 / 234 / 388 | 1.075e-15 |
| 1024 | Banded | ExprLegacy | 11.579 +/- 0.490 | 39.749 +/- 1.882 | 64.162 +/- 2.436 | 393 / 234 / 388 | reference |
| 1024 | Banded | AtomViewNative | 19.706 +/- 0.638 | 41.075 +/- 0.512 | 73.605 +/- 1.429 | 393 / 234 / 388 | 2.541e-16 |
| 2048 | Sparse | ExprLegacy | 25.463 +/- 1.030 | 140.391 +/- 1.724 | 192.310 +/- 2.614 | 404 / 241 / 399 | reference |
| 2048 | Sparse | AtomViewNative | 37.039 +/- 0.741 | 144.055 +/- 3.065 | 206.694 +/- 3.662 | 404 / 241 / 399 | 4.055e-17 |
| 2048 | Banded | ExprLegacy | 25.843 +/- 3.548 | 79.391 +/- 3.600 | 130.700 +/- 6.337 | 404 / 241 / 399 | reference |
| 2048 | Banded | AtomViewNative | 36.681 +/- 0.538 | 82.105 +/- 1.442 | 144.246 +/- 2.006 | 404 / 241 / 399 | 7.698e-17 |

At both dimensions ExprLegacy cold preparation is lower than AtomViewNative by
about 30-41%. Full solve is close: ExprLegacy is about 2.5-3.9% faster in the
reported rows, while identical discrete counters show the methods performed
the same solver work. The largest end-to-end difference is therefore mainly
preparation, not a change in iteration behavior. Debug runs also passed at
`n=128/256/512`; Debug timings are not performance evidence.

A second shared-builder consumer was checked with BDF's fully coupled Dense
workload (`n=32/64/100`). The Debug shared-workload frontend parity test passed
for StiffScalar, Robertson and CombustionLike. In Release, the existing
preparation/stage matrix reached `n=100`: ExprLegacy/AtomViewNative preparation
was `11.617/11.742 ms`, solve `2.502/1.247 ms`, with exact final-state parity
and identical BDF counters. A separate nine-pair telemetry-Off fresh-E2E
capture gave medians ExprLegacy/AtomViewNative of `1.441/1.756 ms` at `n=32`,
`5.799/5.547 ms` at `n=64`, and `12.045/12.527 ms` at `n=100`. The Dense
consumer is mixed and does not establish a universal frontend advantage; its
factorization and telemetry mode dominate small timing deltas.

At this point, the comparison covered current ExprLegacy versus AtomViewNative,
not old versus optimized ExprLegacy inside a solver. That gap is closed by the
direct Release A/B sections below. The frontend comparison remains useful as a
separate route-selection result and must not be conflated with the builder A/B.

## Targeted Release Regression Sweep (2026-10-23)

After the implementation-level A/B and Debug parity gates, targeted Release
regressions were run for the shared-builder consumers and the related IVP
routes:

| Solver | Release gate | Result | Scope note |
| --- | --- | --- | --- |
| LSODE2 | `large_performance_story_tests::lsode2_large_system_sparse_banded_total_and_stage_story`, `n=1024/2048`, 15 repetitions | Passed | ExprLegacy and AtomViewNative had identical work counters and matching trajectories for Sparse and Banded; measurements are recorded above. |
| BDF | `bdf_dense_size_preparation_and_solver_matrix_story` and `bdf_dense_fresh_e2e_routes_alternate_to_control_host_order_bias` | Passed | Shared-workload correctness and alternating ExprLegacy/AtomViewNative Dense solves passed; results are mixed, not a universal speedup claim. |
| Radau | `large_workload_callback_and_full_solve_stage_breakdown`, `RADAU_STORY_STAGE_DIMENSIONS=256,512`, 32 callback repetitions | Passed | ExprLegacy/AtomView and Dense/Sparse/Banded matrix completed with `ok` rows. Radau ExprLegacy uses its own `PreparedSymbolicCallbacks`; this gate is regression coverage, not evidence for the shared ExprLegacy Jacobian builder. |
| BE | `be_symbolic_frontend_selection_preserves_trajectory_and_reports_route` and `be_symbolic_parameter_rebind_matches_fresh_solve_story` | Passed | Frontend trajectory parity, route telemetry and parameter rebind vs fresh solve passed. The latter reported identical trajectories and matching discrete work. |

These were targeted Release gates, not a complete release run of every test in
the four solver modules. The shared builder remains an internal implementation
detail: no public execution-mode switch was added, and the former scalar
implementation is retained only as an isolated test/benchmark reference where
needed. The direct old-vs-new full-solver comparisons are recorded in the
following sections; the earlier frontend comparison is against AtomViewNative.
These targeted solver results do not establish a universal speed claim for
every IVP consumer and do not justify removing the independent Atom frontend.

## Apple-to-Apple LSODE2 Full-Solve A/B (2026-10-23)

To compare the new ExprLegacy preparation with the exact previous
ExprLegacy implementation, a test-only switch replays the former dense scalar
builder: differentiate every equation against every state variable, simplify
the complete dense symbolic Jacobian, then feed it through the same callback
compilation and solver. The switch is available only in `cfg(test)`; it does
not add a runtime or public option. Both routes use identical equations,
parameters, controller, matrix backend, callback policy and initial state.

The Release story used 15 paired repetitions with alternating old/new order
for each pair. Wall times were collected with telemetry Off; a separate
Detailed run checked trajectory and work-counter parity. Reported values are
median/MAD in milliseconds. Solver construction is included in the LSODE2
`total` scope; preparation and integration are timed separately and are not
additive to that outer scope.

| n | layout | route | prepare ms | solve ms | total ms | residual/Jacobian/linear; accepted/rejected | max state diff |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| 256 | Sparse | previous scalar ExprLegacy | 11.022/0.134 | 16.581/0.163 | 30.822/0.244 | 398/237/393; 220/17 | reference |
| 256 | Sparse | optimized ExprLegacy | 2.421/0.070 | 16.477/0.144 | 22.028/0.244 | 398/237/393; 220/17 | 0 |
| 256 | Banded | previous scalar ExprLegacy | 11.004/0.125 | 9.500/0.206 | 23.537/0.198 | 398/237/393; 220/17 | reference |
| 256 | Banded | optimized ExprLegacy | 2.342/0.090 | 9.577/0.281 | 14.954/0.282 | 398/237/393; 220/17 | 0 |
| 512 | Sparse | previous scalar ExprLegacy | 35.889/0.354 | 37.551/0.302 | 79.703/0.730 | 443/275/436; 252/23 | reference |
| 512 | Sparse | optimized ExprLegacy | 4.751/0.087 | 37.608/0.377 | 48.619/0.192 | 443/275/436; 252/23 | 0 |
| 512 | Banded | previous scalar ExprLegacy | 34.757/0.523 | 19.829/0.071 | 60.721/0.464 | 443/275/436; 252/23 | reference |
| 512 | Banded | optimized ExprLegacy | 4.726/0.059 | 20.502/0.156 | 31.363/0.194 | 443/275/436; 252/23 | 0 |

On the final-source rerun, the full solve was `28.5%/36.5%` faster at `n=256`
Sparse/Banded and `39.0%/48.4%` faster at `n=512`. Cold preparation improved
by roughly `4.6-7.6x`; solve time was nearly unchanged, with the largest
difference a `3.4%` Banded increase at `n=512`. Exact discrete work and final
state were preserved. This is direct evidence that the preparation reduction
carries through to complete large LSODE2 solves on both supported sparse
storage routes.

## Apple-to-Apple BDF Dense Full-Solve A/B (2026-10-23)

The same test-only old scalar reference was then used through BDF's
`try_solve`, with the fully coupled Dense workload, identical solver settings,
and alternating old/new order. Solver construction is excluded from the
timed `try_solve` call; measured runs use telemetry Off. Separate Detailed
runs assert exact BDF operation-counter equality and state parity. Values are
median/MAD milliseconds from 15 pairs, except the final `n=256` confirmation,
which uses 31 pairs.

| n | route | `try_solve` median/MAD ms | detailed preparation/solve ms | residual/Jacobian/factor/linear | accepted/rejected; nonlinear solves/iterations | max state diff |
| ---: | --- | ---: | ---: | --- | --- | ---: |
| 32 | previous scalar ExprLegacy | 1.025/0.046 | 1.341/0.387 | 55/1/7/54 | 22/3; 25/54 | reference |
| 32 | optimized ExprLegacy | 1.296/0.072 | 1.613/0.357 | 55/1/7/54 | 22/3; 25/54 | 0 |
| 64 | previous scalar ExprLegacy | 4.830/0.046 | 3.962/1.035 | 55/1/7/54 | 22/3; 25/54 | reference |
| 64 | optimized ExprLegacy | 5.974/0.080 | 6.146/1.020 | 55/1/7/54 | 22/3; 25/54 | 0 |
| 100 | previous scalar ExprLegacy | 14.798/0.178 | 13.178/2.743 | 55/1/7/54 | 22/3; 25/54 | reference |
| 100 | optimized ExprLegacy | 11.795/0.594 | 12.312/2.558 | 55/1/7/54 | 22/3; 25/54 | 0 |
| 128 | previous scalar ExprLegacy | 27.688/0.152 | 23.736/4.787 | 55/1/7/54 | 22/3; 25/54 | reference |
| 128 | optimized ExprLegacy | 20.336/0.842 | 21.947/4.704 | 55/1/7/54 | 22/3; 25/54 | 0 |
| 256 | previous scalar ExprLegacy | 188.900/2.139 | 168.984/20.629 | 55/1/7/54 | 22/3; 25/54 | reference |
| 256 | optimized ExprLegacy | 102.860/1.694 | 96.423/21.478 | 55/1/7/54 | 22/3; 25/54 | 0 |

At the two smallest sizes, the optimized route costs about `0.27 ms` and
`1.14 ms` more per solve; at `n=100/128/256` it is about `20%/27%/45%`
faster. The final-source 31-pair `n=256` confirmation measured `188.900 ms`
versus `102.860 ms` (`45.6%` faster), with about `1.1%` old-route relative MAD
and `1.6%` optimized-route relative MAD. The medium/large gain is consistent with reducing cold
symbolic preparation; the detailed preparation scopes are from separate
instrumented runs and must not be added to uninstrumented `try_solve` time.

## Decision

The apple-to-apple Release evidence now covers large Sparse and Banded LSODE2
and fully coupled Dense BDF. Both show substantial large-system end-to-end
gains with identical solver work and trajectories. BDF retains a small
absolute penalty at `n=32/64`, while LSODE2's tested large cases improve on
both layouts. For the measured workloads this supports keeping the optimized
Expr builder as the production implementation; no public opt-in mode is needed
because the former scalar builder is only a test control. The independent
AtomView frontend remains intact. Radau's separate ExprLegacy callback builder
is not covered by this optimization claim.

## Production Route and Legacy-Code Boundary (2026-10-23)

The production decision applies to the **Expr frontend implementation**, not
to frontend selection: both Expr and native AtomView remain supported. The
public/internal enum name `ExprLegacy` is retained for compatibility, but it
now selects the dependency-aware, batched and size-gated production builder.
Sparse and Banded LSODE2 additionally use their layout-aware coordinate
builders. The old all-columns scalar-differentiation implementation is
compiled only under `cfg(test)` and exists solely for the paired reference
tests above.

The historical LSODE2 AtomView-to-Expr adapter is also declared only under
`cfg(test)` and is used by legacy parity stories. `AtomViewExprCompat` remains
a hidden compatibility/comparison selector; no solver default selects it and
new integrations must use either Expr or native AtomView. Radau is deliberately
outside this migration because its Expr callback builder is a separate active
implementation, not dead shared infrastructure.

Telemetry and failure contracts remain on the active route: preparation
records symbolic differentiation, simplification, Jacobian compilation and
the route/build counters; `try_evaluate_residual` and
`try_evaluate_jacobian` return typed `IvpBackendError` values, including shape
and poisoned-parameter-state failures. The regression test caught and fixed a
previously unguarded Expr callback input shape that could panic inside
Lambdify; the argument binder now returns `InvalidStateShape` before closure
evaluation and records the error in telemetry. The regression
`optimized_expr_route_reports_typed_callback_errors_and_stage_telemetry`
checks those guarantees on ExprLegacy, while the LSODE2/BDF stories verify
identical discrete solver work and trajectories. Legacy infallible callback
closures are retained only as compatibility adapters; new internal consumers
should use the fallible methods.

## AtomView IVP Evaluator Compilation A/B (2026-10-23)

An isolated candidate compares serial `PreparedEvaluator::new_with_context`
with deterministic indexed Rayon compilation for the same already-differentiated
Atom sparse Jacobian. Differentiation is outside the timed region. A Debug
correctness test first checked a 48-state cyclic nonlinear system; row/column
order and evaluated values matched exactly.

| n | Jacobian entries | sequential median/MAD ms | four-worker median/MAD ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 768 | 0.269/0.009 | 0.131/0.009 | 2.06x |
| 1024 | 3072 | 1.096/0.022 | 0.567/0.102 | 1.93x |
| 2048 | 6144 | 2.134/0.055 | 1.136/0.032 | 1.88x |

This confirms a useful isolated preparation-stage reduction, with more than
`1 ms` saved at `n=2048`. It does not yet prove a full-solve win: these rows
exclude Expr-to-Atom conversion, differentiation, solver setup and integration.
Production compilation therefore remains sequential pending a paired LSODE2
full-solve A/B with telemetry-Off wall time, exact work/trajectory parity,
and allocation/retained-memory measurements. Reproduce the component test with
`IVP_ATOM_COMPILE_AB_DIMENSIONS=256,1024,2048`,
`IVP_ATOM_COMPILE_AB_REPETITIONS=31`, and
`cargo test --release --lib atomview_evaluator_compile_sequential_parallel_release_ab -- --ignored --nocapture`.

### LSODE2 Full-Solve Gate

The solver-level gate uses the same AtomView frontend, sequential callback
execution, equations, initial state and solver settings for both candidates.
Only compilation of the already-differentiated Jacobian evaluators changes.
Correctness/counter checks use separate detailed-telemetry runs; timing pairs
are alternating, with telemetry Off. The parallel candidate uses four Rayon
workers. The selector and candidate branch exist only in test builds; the
production route remains sequential.

| matrix | n | serial prepare/solve/total median/MAD ms | parallel prepare/solve/total median/MAD ms | total change | final-state diff | counters (res/jac/linear/accepted/rejected) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Sparse | 256 | 4.943/0.113 ; 17.487/0.127 ; 25.506/0.133 | 4.668/0.090 ; 17.464/0.277 ; 25.411/0.332 | -0.4% | 0 | 398/237/393/220/17 |
| Banded | 256 | 4.671/0.049 ; 10.008/0.049 ; 17.773/0.149 | 4.554/0.093 ; 10.194/0.209 ; 17.880/0.186 | +0.6% | 0 | 398/237/393/220/17 |
| Sparse | 1024 | 18.819/0.847 ; 68.670/2.096 ; 100.686/3.892 | 18.084/0.975 ; 69.412/1.277 ; 101.631/2.572 | +0.9% | 0 | 393/234/388/217/17 |
| Banded | 1024 | 19.672/0.854 ; 41.355/0.410 ; 75.059/1.576 | 18.563/0.462 ; 41.987/0.788 ; 74.142/1.135 | -1.2% | 0 | 393/234/388/217/17 |
| Sparse | 2048 | 39.009/3.441 ; 141.731/5.704 ; 205.727/10.588 | 38.569/3.547 ; 143.298/8.556 ; 205.041/12.595 | -0.3% | 0 | 404/241/399/222/19 |
| Banded | 2048 | 39.383/1.725 ; 82.739/2.799 ; 151.875/8.250 | 38.171/1.713 ; 88.427/4.943 ; 153.658/9.939 | +1.2% | 0 | 404/241/399/222/19 |

The isolated evaluator-compiler speedup (about `1.9x`) saves only `0.4–1.2 ms`
of preparation on the large solver cases. Full-solve changes range from a
`1.2%` apparent gain to a `1.2%` regression, with differences inside the
observed MAD and no consistent advantage across layouts or sizes. This is not
a production optimization: keep serial evaluator compilation. Debug parity
also passed at `n=32` for both layouts with exact final-state equality and
matching discrete counters.

Reproduce the release gate with
`RAYON_NUM_THREADS=4`,
`LSODE2_ATOM_COMPILE_AB_DIMENSIONS=256,1024`,
`LSODE2_ATOM_COMPILE_AB_REPETITIONS=15`, and
`cargo test --release --lib numerical::LSODE2::large_performance_story_tests::lsode2_atomview_evaluator_compile_full_solve_ab -- --ignored --nocapture`.
The large confirmation used `LSODE2_ATOM_COMPILE_AB_DIMENSIONS=2048` and
`LSODE2_ATOM_COMPILE_AB_REPETITIONS=31` with the same command.

## LSODE2 Shared Atom-Graph A/B (2026-10-23)

The ordinary LSODE2 AtomView/Lambdify preparation converted the same residual
equations from Expr once for the residual callbacks and again for the native
Jacobian. The test-only candidate reuses the residual's immutable `Arc<[Atom]>`
when preparing that Jacobian. This avoids one conversion of every equation;
symbolic differentiation, evaluator compilation, callbacks and solver settings
are otherwise unchanged. The production route remains unchanged while
lifecycle/rebind coverage is completed.

Debug parity at `n=32` passed for Sparse and Banded. Both candidates had exact
final-state equality and identical solver counters; the detailed telemetry
reported two `ExprToAtom` stage calls for the separate-graph route and one for
the shared graph.

Release measurements used 31 alternating pairs, four Rayon workers and
telemetry Off during timed runs. Entries show median milliseconds:

| layout | n | prepare serial -> shared | full solve serial -> shared | total improvement |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 256 | 4.914 -> 3.405 | 25.565 -> 23.828 | 6.8% |
| Banded | 256 | 4.837 -> 3.313 | 18.047 -> 16.508 | 8.5% |
| Sparse | 1024 | 19.681 -> 13.513 | 102.465 -> 96.226 | 6.1% |
| Banded | 1024 | 19.104 -> 13.315 | 73.988 -> 67.710 | 8.5% |
| Sparse | 2048 | 38.008 -> 26.067 | 206.972 -> 194.256 | 6.1% |
| Banded | 2048 | 37.899 -> 25.796 | 146.417 -> 134.915 | 7.9% |

At `n=2048`, sharing saved `12.7 ms` per Sparse solve and `11.5 ms` per
Banded solve. Final-state differences were exactly zero. Residual/Jacobian/
linear-solve and accepted/rejected-step counts matched exactly; at `n=2048`
these were `404/241/399/222/19` for each route. The prepare reduction was
about `30–32%` across all tested sizes, and full-solve reductions were
consistent across both matrix layouts. A separate Debug lifecycle test now
covers parameter rebind and structural reconfiguration for Sparse and Banded
against fresh solver oracles. Trajectories and per-solve discrete counts match
exactly; rebind retains the graph, while reconfiguration builds a fresh graph.
This is a promising optimization rather than a microbenchmark-only result, but
it remains a test-only candidate pending targeted LSODE2 regression runs and an
explicit production-promotion decision.

A second 31-pair Release confirmation, with four Rayon workers, measured at
`n=2048` prepare/total medians of `34.747/191.406 -> 23.438/180.263 ms` for
Sparse and `34.807/135.991 -> 23.288/124.358 ms` for Banded. The complete solve
improved by `5.8%` and `8.6%`; at `n=1024`, gains were `5.8%` Sparse and
`10.4%` Banded. Numerical outputs and all discrete work counts remained
identical. Shared graph reuse is now the production LSODE2 AtomView Lambdify
route whenever the residual retains its immutable Atom graph. Generated
residual routes without a retained graph continue to use the established
conversion path, and AOT behavior is unchanged. The separate-graph path remains
test-only as the A/B reference. A post-promotion Release run with 31 alternating
pairs confirmed the effect: at `n=2048`, Sparse total time was `212.084 ->
197.679 ms` (`6.8%` faster) and Banded was `151.376 -> 138.025 ms` (`8.8%`
faster); preparation fell by about one third. Outputs and discrete work counts
were identical. The LSODE2 story contains the full size/layout table and MADs.

Reproduce with `RAYON_NUM_THREADS=4`,
`LSODE2_ATOM_GRAPH_AB_DIMENSIONS=256,1024,2048`,
`LSODE2_ATOM_GRAPH_AB_REPETITIONS=31`, and
`cargo test --release --lib numerical::LSODE2::large_performance_story_tests::lsode2_atomview_shared_residual_graph_full_solve_ab -- --ignored --nocapture`.
