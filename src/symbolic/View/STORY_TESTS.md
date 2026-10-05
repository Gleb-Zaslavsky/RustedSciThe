# Symbolic View Story Tests

This file owns low-level Atom/Expr operation-form diagnostics. Solver-specific
stories in LSODE2 and BVP consume these results through real Jacobian and
callback fixtures, but do not own the corpus.

## Operation Lowering Micro-Corpus

- Test: `symbolic::View::operation_corpus_tests::operation_lowering_micro_corpus_preserves_values_and_reports_shape`
- Date: 2026-09-23
- Scope: debug structural and evaluator attribution; no production lowering change
- Command:

```powershell
cargo test --lib --no-default-features symbolic::View::operation_corpus_tests::operation_lowering_micro_corpus_preserves_values_and_reports_shape -- --ignored --nocapture --test-threads=1
```

- Report:
  `test_reports/Symbolic_View/symbolic__View__operation_lowering_micro_corpus_preserves_values_and_reports_shape.md`
- Result: 22 operation forms passed numerical parity with `max_diff <= 1e-12`.
- Routes: direct `Expr` closure, `Atom -> Expr` compatibility closure, and direct `AtomNative` prepared closure.
- Interpretation: operation shape matters, but node count alone does not predict evaluator cost. Division/reciprocal and function-heavy forms remain candidates for targeted lowering work; subtraction chains, n-ary forms and repeated subexpressions can improve.
- Caveat: the scalar native closure includes prepared-node dispatch and thread-local workspace access. Debug timings are attribution data only and must not replace release measurements on real LSODE2/BVP fixtures.
- Next: compare the three boundaries on frozen production-sized Jacobian entries, separating preparation, closure construction, argument binding, evaluator execution and output assembly.

The production-sized integration consumer is kept in LSODE2 because it owns
the real Jacobian fixtures. Its test name is
`numerical::LSODE2::story_tests2::lsode2_view_three_boundary_real_jacobian_release_story`.
The View corpus remains the owner of operation-level diagnostics; LSODE2 only
supplies the larger workload and solver-facing parity boundary.

## Scalar Jacobian Lowering Corpus

- Test: `symbolic::View::scalar_jacobian_story_tests::scalar_jacobian_lowering_story_preserves_values_and_reports_stages`
- Date: 2026-09-23 (migration; release rerun pending)
- Scope: four scalar Jacobian expressions, no matrix assembly or solver loop.
- Command:

```powershell
cargo test --release --lib --no-default-features symbolic::View::scalar_jacobian_story_tests::scalar_jacobian_lowering_story_preserves_values_and_reports_stages -- --ignored --nocapture --test-threads=1
```

- Report: `test_reports/Symbolic_View/symbolic__View__scalar_jacobian_lowering_story_preserves_values_and_reports_stages.md`
- Historical LSODE2 report migrated on 2026-09-23:
  `test_reports/Symbolic_View/historical__lsode2_atomview_exprcompat_scalar_expression_shape_corpus_story.md`
- Routes: `ExprLegacy`, `AtomViewExprCompat` and `AtomNative`.
- Corpus: exponential/integer power, fractional power with exponential,
  nested fractional power, and transcendental rational expression.
- Required evidence: numerical parity, input/output shape, preparation and
  closure construction time, and repeated scalar evaluation time.

The test supersedes the former LSODE2-owned scalar expression-shape corpus.
Its historical release observation is preserved here as migration evidence:

```text
case                       | historical nodes | ExprLegacy nodes | Compat nodes | historical ns | ExprLegacy ns | Compat ns
------------------------------------------------------------------------------------------------------------------------
exp_times_integer_power    |              64 |               63 |           64 |       245.355 |       257.090 |    250.600
fractional_power_times_exp |              62 |               58 |           62 |       228.460 |       212.245 |    238.575
nested_fractional_power    |              68 |               71 |           68 |       274.810 |       294.190 |    275.485
transcendental_rational    |              65 |               56 |           65 |       270.390 |       231.870 |    264.075
```

Historical operation fingerprints showed that AtomView compatibility and the
old AtomView route had the same shape while ExprLegacy used a different form:

```text
case                       | route                 | nodes | repeated | div | pow | pow-1 | pow-frac | functions
----------------------------------------------------------------------------------------------------------------
exp_times_integer_power    | historical-AtomView   |    64 |       35 |   0 |   9 |     0 |        3 | exp:3
exp_times_integer_power    | ExprLegacy            |    63 |       34 |   2 |   9 |     0 |        3 | exp:2
fractional_power_times_exp | historical-AtomView   |    62 |       34 |   0 |   9 |     3 |        3 | exp:3
fractional_power_times_exp | ExprLegacy            |    58 |       30 |   2 |   7 |     0 |        2 | exp:3
nested_fractional_power    | historical-AtomView   |    68 |       36 |   0 |  13 |     0 |        7 | exp:3
nested_fractional_power    | ExprLegacy            |    71 |       39 |   2 |  14 |     0 |        6 | exp:2
transcendental_rational    | historical-AtomView   |    65 |       35 |   0 |  10 |     3 |        0 | cos:1,exp:3,sin:2
transcendental_rational    | ExprLegacy            |    56 |       30 |   2 |   7 |     0 |        0 | cos:1,exp:3,sin:1
```

These historical rows are diagnostic evidence, not a current baseline. The
new View-owned report must be rerun in release before drawing conclusions.

## Production-Sized AtomNative Regression Gate: 2026-09-24 01:27 Local

The latest LSODE2 release consumer remains the source of the real Jacobian
fixture, while this document records the low-level View conclusion. The full
report is:

`test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_view_three_boundary_real_jacobian_release_story.md`

The `diffusion-chain` case is a mandatory AtomNative regression gate:

```text
route       | nonzero | symbolic_ms | atom_convert_ms | closure_ms | eval_ns/call
ExprLegacy  |     382 |       1.057 |           0.000 |      0.027 |       488.950
AtomNative  |     382 |       0.000 |           0.168 |      6.320 |      1928.150
```

The result is numerically correct but approximately `3.94x` slower in the
repeated native callback measurement. This is a real workload-specific gate,
not a small-sample fluctuation to discard. It must be reproduced after every
native evaluator or argument-binding change, together with the three-body case
where AtomNative is faster. The two workloads prevent optimizing for one tree
shape while regressing another.

The current hypothesis is deliberately narrower than "Atom has too many
nodes": the diffusion-chain routes have the same structural shape. The next
measurements must separate parameter/state binding, prepared-node dispatch,
thread-local workspace access and scalar evaluation. No canonical tree rewrite
is accepted without parity and both real-fixture gates.
