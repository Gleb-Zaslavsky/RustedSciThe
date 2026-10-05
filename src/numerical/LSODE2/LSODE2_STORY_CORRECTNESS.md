# LSODE2 Correctness Stories

Correctness is the first gate for every Lambdify and AOT change. The corpus
checks residual and Jacobian values, sparse ordering, compact-Banded slots,
parameter binding, non-finite behavior, typed shape errors and solver
trajectory parity.

## Required Gates

- `correctness_story_tests`: debug parity, invalidation, failure injection,
  sparse order, band slots and non-finite values.
- `aot_trajectory_parity_story_tests`: accepted/rejected steps, Jacobian
  refreshes, linear solves and final trajectory across AOT/Lambdify routes.
- `parameter_continuation_story_tests`: fresh-solver oracle for numeric
  rebinding without structural rebuild.
- `aot_layout_parity_story_tests`: Sparse/Banded output layout and chunk
  policy parity.

All release baselines must print integer counters next to timing rows. A
roundoff-level final-state difference is acceptable only when the route's
trajectory and callback counters also match.

## Open Safety Work

Schema, mesh/layout, boundary and Jacobian-pattern changes must invalidate a
prepared runtime and reject stale `RequirePrebuilt` artifacts. Numeric-only
parameter changes may reuse prepared callbacks, but must not reuse stale
factors or bindings.

See the immutable [historical correctness record](LSODE2_STORY_ARCHIVE.md)
and the current checklist in [TODO.md](TODO.md).

## 2026-09-29 Release Verification

All completed release correctness and lifecycle reports from the post-fix sweep
passed. The release gates cover AOT trajectory parity, Sparse/Banded layout
parity, numeric parameter rebind, non-finite and typed-shape failures,
producer/consumer artifact reuse, schema/layout/Jacobian-pattern invalidation,
compact-Banded ExprLegacy control and repeated `RequirePrebuilt` solves.

The long Criterion parameter-continuation benchmark is still pending; this is
not a correctness gap. Its story-level fresh-solver comparisons already report
zero state/time difference and matching counters for ExprLegacy and
AtomViewNative on both layouts.
