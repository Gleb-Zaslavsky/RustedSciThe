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
- Routes: direct `Expr` closure, `Atom -> Expr` compatibility closure, direct `AtomNative` Lambdify closure, and explicit `PreparedEvaluator` with caller-owned scratch.
- Interpretation: operation shape matters, but node count alone does not predict evaluator cost. Division/reciprocal and function-heavy forms remain candidates for targeted lowering work; subtraction chains, n-ary forms and repeated subexpressions can improve.
- Caveat: the scalar native closure includes prepared-node dispatch and thread-local workspace access. Debug timings are attribution data only and must not replace release measurements on real LSODE2/BVP fixtures.
- Next: compare all four execution boundaries on frozen production-sized Jacobian entries, separating preparation, closure construction, argument binding, evaluator execution and output assembly.

### Explicit PreparedEvaluator Debug And Benchmark Coverage (2026-10-22)

- Correctness test: `symbolic::View::operation_corpus_tests::operation_lowering_micro_corpus_preserves_values_and_reports_shape`.
- Command:

```powershell
cargo test --lib operation_lowering_micro_corpus_preserves_values_and_reports_shape -- --ignored --nocapture
```

- Result: all 22 scalar forms preserve value parity across Expr Lambdify,
  Atom-to-Expr compatibility, AtomNative Lambdify and direct
  `PreparedEvaluator::evaluate_with_scratch_typed`.
- The explicit prepared route obtains its ordered symbols from
  `PreparedSparseAtomSystem::from_atoms_discovering_dependencies`; it does not
  synthesize symbol identities independently.
- Lifecycle test:
  `symbolic::View::prepared::tests::prepared_runtime_large_sparse_rebind_series_preserves_values_and_capacity`.
- Command:

```powershell
cargo test --lib prepared_runtime_large_sparse_rebind_series_preserves_values_and_capacity -- --nocapture
```

- Result: a 128-equation sparse chain with 255 Jacobian entries passed 256
  rebind cycles against independent analytic residual/Jacobian formulas. The
  warmed scratch capacity stayed constant. This checks retained vector/hash
  capacity only; it is not an allocation counter or a complete memory bound.
- Worker-ownership test:
  `symbolic::View::prepared::tests::prepared_runtime_shared_plan_uses_worker_owned_scratch_concurrently`.
  Four concurrent workers shared the immutable prepared plan and kept
  independent scratch and output buffers through 64 rebinds each. The test
  computes a sequential reference for those same 4 x 64 inputs, then requires
  exact residual/Jacobian buffer equality from concurrent calls; independent
  analytic formulas are checked as well. This is a correctness check, not a
  parallel speedup claim. `PreparedAtomSystem` has no execution-policy setting,
  so this validates caller-managed concurrency.
- Parameter/input ABI test:
  `symbolic::View::prepared::tests::prepared_runtime_parameterized_abi_matches_analytic_and_finite_difference_jacobian`.
  A four-input system in `[y, q, x, p]` order passed at three distinct states.
  Residuals matched independent formulas; every stored Jacobian value matched
  analytic derivatives and central finite differences, and the structural zero
  was absent from the sparse stream. This is a synthetic contract check.
  Production callback and trajectory parity is separately recorded below for
  LSODE2 and BVP_Damp; it does not imply that the standalone
  `PreparedAtomSystem` has been integrated into those solvers.
- Criterion target: `benches/symbolic_prepared_runtime.rs`, registered as
  `symbolic_prepared_runtime`. It uses the same sparse-chain residuals and
  Jacobian entries at dimensions 128 and 512, checks route parity before
  timing, and compares Expr closures, Atom Lambdify closures and explicit
  caller-scratch evaluation. Preparation and solver/linear-algebra work are
  excluded; output buffers are reused for every route.
- Compile check passed with `cargo check --bench symbolic_prepared_runtime`.
  No Release timing has been recorded yet. Run it after the current symbolic
  A/B batch is ready, using:

```powershell
cargo bench --bench symbolic_prepared_runtime -- --noplot
```

- Interpretation: explicit scratch is not assumed to beat closures. Atom
  Lambdify already uses the same prepared evaluator with thread-local scratch;
  this benchmark tests whether caller-owned scratch changes warm callback
  throughput under a larger batch. Debug timings remain diagnostic only.

#### Differentiation Pipeline Parity (2026-10-22)

- Test: `symbolic::View::prepared::tests::expr_atom_derivative_normalization_and_prepared_jacobian_are_componentwise_equal`.
- Command:

```powershell
cargo test --lib --no-default-features expr_atom_derivative_normalization_and_prepared_jacobian_are_componentwise_equal -- --nocapture --test-threads=1
```

- Result: five representative residual forms covering subtraction, negative
  and fractional powers, functions, variable powers and widely scaled
  coefficients were differentiated with respect to three ABI-ordered inputs
  at three in-domain states. `Expr::diff`, converted/normalized Atom
  derivatives, Atom-to-Expr round-trips and `PreparedAtomSystem` sparse
  Jacobian entries agree componentwise, including omitted structural zeros.
  This is a correctness gate, not a performance result. The full operation
  corpus is now covered by the next gate below.

#### Full Operation-Corpus Derivative Parity (2026-10-22)

- Test: `symbolic::View::operation_corpus_tests::operation_corpus_derivative_pipeline_preserves_componentwise_parity`.
- Command:

```powershell
cargo test --lib --no-default-features operation_corpus_derivative_pipeline_preserves_componentwise_parity -- --nocapture --test-threads=1
```

- Result: all 22 operation-corpus expressions were checked for residual and
  derivative parity with respect to `x`, `y` and `p` at three finite,
  in-domain states. The test compares Expr differentiation/evaluation,
  Expr-to-Atom conversion, normalized Atom differentiation/evaluation,
  Atom-to-Expr round-trip, `PreparedAtomSystem` prepared from both Expr and
  direct Atom inputs, and its sparse Jacobian output; omitted structural
  entries are checked as zero.
  This closes parity for the maintained View operation corpus, not every
  public function or solver-integration combination. No production evaluator
  behavior or performance claim is implied.

#### Built-In Unary Function Derivative Parity (2026-10-22)

- Test: `symbolic::View::operation_corpus_tests::every_builtin_unary_function_preserves_expr_atom_derivative_parity`.
- Command:

```powershell
cargo test --lib --no-default-features every_builtin_unary_function_preserves_expr_atom_derivative_parity -- --nocapture --test-threads=1
```

- Result: all ten built-in unary Expr functions (`exp`, `log`, `sin`, `cos`,
  `tan`, `cot`, `asin`, `acos`, `atan`, `acot`) passed residual and derivative
  comparisons at three finite points inside the real-domain intersection.
  Compared routes are direct Expr, direct AtomNative, Atom-to-Expr round-trip,
  and the `PreparedAtomSystem` residual/Jacobian. The test also asserts the
  expected single structural Jacobian entry for each unary function.
- Regression found and fixed: Atom differentiation of `acos(x)` constructed
  `(x^2 - 1)^(-1/2) * -1`, which is NaN throughout the valid real interior.
  It now constructs `(1 - x^2)^(-1/2) * -1`, matching the Expr derivative.
  The exact derivative-tree expectation in `extended_trig_derivatives` was
  updated to assert the corrected formula.
- Scope: this closes parity for the built-in unary constructors, alongside
  arithmetic/power coverage in the maintained 22-form corpus. It does not
  cover custom Atom functions, unsupported-operation error exhaustiveness or
  compatibility IEEE versus typed-domain policy. No production or performance
  claim is implied.

#### Prepared Sparse Structure Edge Cases (2026-10-22)

- Tests: `symbolic::View::prepared::tests::prepared_runtime_preserves_sparse_structure_across_zero_and_empty_rows` and `symbolic::View::prepared::tests::prepared_runtime_asymmetric_bandwidth_keeps_expected_row_major_entries`.
- Command:

```powershell
cargo test --lib --no-default-features symbolic::View::prepared::tests -- --nocapture --test-threads=1
```

- Result: the parameterized Jacobian retains its row-major structure as a
  coefficient moves through positive, exact zero and negative values; an
  empty equation and unused input column add no entries. Diagonal-only and
  asymmetric `(lower, upper)` bandwidth cases preserve the documented ordered
  entries and values. These are contracts of the generic sparse prepared
  runtime; endpoint/boundary rows and Dense/Banded storage adapters remain
  consumer-level integration gates.

#### Typed Scratch Failure And Recovery

- Test: `symbolic::View::evaluate::test::caller_owned_scratch_keeps_typed_domain_policy_and_recovers_after_error`.
- Command:

```powershell
cargo test --lib caller_owned_scratch_keeps_typed_domain_policy_and_recovers_after_error -- --nocapture
```

- Result: for `log(-1)`, Atom Lambdify preserves its compatibility IEEE result
  (`NaN`), while the explicit typed scratch call returns
  `PreparedEvaluationError::Domain`. Reusing that same scratch at `x=4` then
  succeeds and agrees with the closure. The distinction is intentional; the
  warm callback benchmark uses finite in-domain inputs and does not compare
  these different failure policies as if they were equivalent.

The production-sized integration consumer is kept in LSODE2 because it owns
the real Jacobian fixtures. Its test name is
`numerical::LSODE2::story_tests2::lsode2_view_three_boundary_real_jacobian_release_story`.
The View corpus remains the owner of operation-level diagnostics; LSODE2 only
supplies the larger workload and solver-facing parity boundary.

### LSODE2 Combustion Componentwise View Correctness (2026-10-22)

- Test:
  `numerical::LSODE2::legacy_story_impl::legacy_story_view::lsode2_combustion_view_componentwise_parity_across_states_and_parameters`.
- Command:

```powershell
cargo test --lib lsode2_combustion_view_componentwise_parity_across_states_and_parameters -- --nocapture
```

- Result: the canonical `combustion_like_story_base_config()` RHS passed at
  three state vectors and three parameter bindings. Direct prepared Atom
  residuals matched Expr RHS evaluation; state-Jacobian sparsity and values
  matched ExprLegacy; each state derivative also matched central differences
  of the original RHS. The actual LSODE2 Sparse and Banded callbacks were also
  compared across ExprLegacy and direct AtomView, and every matrix coordinate
  agreed with the generic prepared Atom Jacobian. This covers sparse triplet
  ordering and the production compact band-slot mapping on the same bindings.
- Fixture guard: the test pins `fnv1a64-2b72ad61b128b0a2` over canonical
  equation text, variable/time ABI, parameter names and exact initial/parameter
  `f64` bit patterns. A change to the compiler input or its textual form must
  be reviewed explicitly rather than silently changing the acceptance case.
- Scope: this is a debug correctness gate, not an integration trajectory or
  performance measurement. The fixture has the canonical three state
  variables and seven equation parameters. The View-level gate still needs
  captured accepted trajectory states, frozen fixture/config hashes and BVP
  fixtures; the current Sparse/Banded slot check covers the LSODE2 callback
  fixture only.

### BVP_sci Combustion Componentwise View Correctness (2026-10-22)

- Test: `numerical::BVP_sci::new::story_tests::combustion_bvp_componentwise_frontend_parity_pins_compiler_input`.
- Command:

```powershell
cargo test --lib combustion_bvp_componentwise_frontend_parity_pins_compiler_input -- --nocapture
```

- Result: the six-state combustion-shaped BVP_sci callback fixture passed in
  Debug. ExprLegacy and AtomViewNative residuals matched Expr Lambdify; their
  fixed sparse Jacobian coordinate order matched (12 entries); sparse values
  agreed with dense output; frontend Jacobians agreed; and each AtomView state
  derivative matched a central finite difference.
- Fixture guard: `fnv1a64-163ad2c0db7000c1` covers canonical equation text,
  state/time ABI and exact evaluation-point/state `f64` bits. A printer or
  fixture change requires deliberate review.
- Scope: pointwise callback correctness only. The test does not establish
  trajectory-state parity, large-system behavior or performance; frozen BVP_sci
  trajectory states, larger BVP_Damp combustion meshes and additional fixtures
  remain open.

### BVP_Damp Combustion Callback Componentwise View Correctness (2026-10-22)

- Test: `numerical::BVP_Damp::test_backend_compare::tests::combustion_fixture_componentwise_callbacks_match_frontends_and_matrix_layouts`.
- Command:

```powershell
cargo test --lib combustion_fixture_componentwise_callbacks_match_frontends_and_matrix_layouts -- --nocapture
```

- Result: the production six-equation combustion RHS passed in Debug at the
  eight-node initial guess. ExprLegacy and AtomView matched on Sparse-faer and
  native Banded residual/Jacobian callbacks; the ExprLegacy/Sparse Jacobian
  matched central finite differences across every global state column.
- Fixture guard: `fnv1a64-07846b22e620f6fa` covers canonical equations, state/
  time/parameter ABI, interval, mesh, boundary conditions and initial-guess
  `f64` bits.
- Scope: callback correctness, not accepted-trajectory parity, large-mesh
  behavior or performance. Those remain separate gates.

### BVP_Damp Accepted-Trajectory Callback Parity (2026-10-22)

- Test: `numerical::BVP_Damp::test_lambdify_cross_product::tests::accepted_trajectory_callbacks_match_on_medium_sparse_and_banded_meshes`.
- Command:

```powershell
cargo test --lib accepted_trajectory_callbacks_match_on_medium_sparse_and_banded_meshes -- --nocapture
```

- Result: passed in Debug for a 128-interval oscillator BVP. Sparse-faer and
  native Banded with ExprLegacy and AtomView each solved the same problem; all
  four callback routes were then evaluated at the same accepted reduced Newton
  vector. Residuals and Jacobians matched, and every reference-Jacobian column
  matched central finite differences. The public postprocessed trajectory was
  separately confirmed to have shape `129 x 2`.
- ABI note: `get_result()` reinserts fixed boundary values and is therefore not
  a callback input. The test deliberately uses the solver's accepted reduced
  Newton vector, which matches the callback dimension.
- Scope: medium-mesh accepted-state parity, not a frozen trajectory corpus,
  large combustion workload or performance measurement.

### BVP_Damp Combustion-1000 Compact Jacobian-Action Parity (2026-10-22)

- Test: `numerical::BVP_Damp::test_backend_compare::tests::combustion_1000_mesh_jacobian_actions_match_frontends_and_layouts`.
- Command:

```powershell
cargo test --lib combustion_1000_mesh_jacobian_actions_match_frontends_and_layouts -- --nocapture --test-threads=1
```

- Result: passed in Debug for the production six-equation combustion fixture
  at 1000 intervals. ExprLegacy/AtomView and Sparse-faer/native Banded residuals
  and `J·v` values matched at a common canonical input; reference `J·v` matched
  a central directional finite difference.
- Fixture guard: `fnv1a64-efdb92ea2e1d64fb` covers the fixed problem ABI and
  exact initial-guess `f64` bits.
- Memory/semantic note: matrices were retained in native compact forms and
  compared through matrix-vector products; this avoids dense materialization
  and does not claim parity of accepted combustion trajectories or solver
  iteration traces.

### BVP_Damp Combustion-1000 Accepted-Trajectory Parity (2026-10-22)

- Test: `numerical::BVP_Damp::test_backend_compare::tests::combustion_1000_accepted_trajectory_matches_frontends_and_layouts`.
- Command:

```powershell
cargo test --lib combustion_1000_accepted_trajectory_matches_frontends_and_layouts -- --nocapture --test-threads=1
```

- Result: all four ExprLegacy/AtomView x Sparse-faer/native Banded Lambdify
  solves converged. The complete `1001 x 6` postprocessed trajectories matched
  componentwise; maximum difference from ExprLegacy/Sparse was `7.878246666e-15`
  (limit `1e-6`). Seven reference node rows are frozen in the test with
  component tolerance `1e-8`.
- Scope: accepted trajectory parity for this combustion fixture only. Newton
  counters and semantic decision events also matched exactly across all four
  routes: 5 iterations, 12 residual requests, 1 Jacobian recalculation, 5
  damping trials, 0 rejections, 10 linear solves, 0 refinements and 7 selected
  decision events. This workload itself exercises no rejection/refinement;
  AOT routes are not covered. No performance claim is made.

### BVP_Damp Newton And Refinement Trace Corpus (2026-10-22)

- Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_parity_corpus -- --nocapture --test-threads=1
```

- Result: all six Debug tests passed. The real symbolic Bratu-like fixture
  matched ExprLegacy/AtomView at 6 Newton iterations, 6 damping trials, zero
  rejections, one grid refinement and identical typed event traces. A separate
  deterministic fixture matched 13 iterations, 72 damping trials and 64
  rejections after replacing both prepared routes with the same numeric
  callbacks; it proves solver decision parity, not symbolic callback parity.
- Scope: real-symbolic refinement trace now matches across Sparse/Banded and
  ExprLegacy/AtomView. AOT-vs-Lambdify trajectory and Newton-decision parity
  also pass on both layouts, with telemetry confirming `AotCompiled`. The
  controlled rejected-step test matches all four frontend/layout settings and
  uses route-native Sparse/Banded Jacobians, but replaces callbacks after
  symbolic preparation. It therefore does not prove rejected-step parity when
  the symbolic RHS/Jacobian themselves produce the rejected trajectory.

## Small finite f64 coefficient fidelity

- Date: 2026-10-19
- Scope: debug-only Expr-to-AtomView numerical fidelity; no nonlinear solver,
  release benchmark, or performance conclusion.
- Tests:
  `symbolic::View::atomview_diagnostics::tests::diagnostic_reports_small_constant_conversion_without_solver_involvement`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_reports_small_coefficient_inside_power_expression`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_sweep_covers_both_signs_near_observed_boundary`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_preserves_small_f64_during_packed_numeric_arithmetic`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_does_not_panic_or_erase_overflowed_float_power`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_preserves_subnormal_and_minimum_normal_constants`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_preserves_maximum_finite_and_mixed_magnitudes`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_preserves_cancellation_and_small_terms_in_compositions`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_covers_signed_integer_fractional_and_reciprocal_powers`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_covers_zero_signed_zero_and_variable_exponent_edges`,
  `symbolic::View::atomview_diagnostics::tests::diagnostic_covers_function_boundaries_and_nonfinite_classification`.
- Command:

```powershell
cargo test --lib --no-default-features --locked symbolic::View::atomview_diagnostics -- --nocapture --test-threads=1
```

- Initial diagnosis: the old finite-f64 fallback attempted a compact rational
  with a fixed scale of `1e6`. Values below that scale could become a zero
  numerator after the rational candidates were rejected, before AtomView
  evaluation began.
- Repair: finite values that do not fit the compact rational representation
  are stored in an internal packed f64 numeric leaf and are handled by the
  AtomView evaluator, printer, conversion and code-generation paths. Numeric
  powers use f64 folding when finite; exact or float overflow remains a
  symbolic power instead of panicking or becoming zero.
- 2026-10-22 extension: the complete 11-test debug diagnostic suite passes.
  Added explicit `x^0`, `x^1`, `x^-1` and variable `x^y` cases over positive,
  negative, `+0.0` and `-0.0` bases, plus reciprocal behavior at minimum-normal
  and maximum finite magnitudes. Expr and AtomView agree on finite values,
  signed zero, NaN/infinity classification and infinity sign. This adds
  correctness coverage only; it is not a performance result.
- Result: `10 passed; 0 failed`; the coefficient `+-5.09024475e-16`, the
  composed `c*T^4` case at `T=2500`, and both-sign sweeps remain nonzero and
  agree with the original f64 value within the diagnostic tolerance. The
  extended matrix also covers subnormal and maximum finite values, mixed
  magnitudes, cancellation, signed/fractional/reciprocal powers, function
  boundaries, and matching `NaN`/infinity classification.
- Remaining boundary: direct textual View parsing still has a separate exact
  rational grammar. Its audit is now covered by the parser tests below: exact
  scientific literals stay rational when possible, finite long literals use
  packed f64, and nonfinite/incomplete literals fail explicitly.

## Text parser scientific-notation fidelity

- Date: 2026-10-19
- Scope: debug parser correctness; no solver or performance measurement.
- Tests:
  `symbolic::View::parser::test::parses_scientific_notation_exactly`,
  `symbolic::View::parser::test::parses_uppercase_scientific_notation`,
  `symbolic::View::parser::test::preserves_finite_scientific_literal_beyond_exact_rational_range`,
  `symbolic::View::parser::test::preserves_long_scientific_literal_inside_composed_expression`,
  `symbolic::View::parser::test::rejects_nonfinite_scientific_literal_instead_of_encoding_it`.
- Command:

```powershell
cargo test --lib --no-default-features --locked symbolic::View::parser::test -- --nocapture --test-threads=1
```

- Result: `13 passed; 0 failed`.
- Contract: `e/E` notation is accepted; compact exact rationals remain exact;
  finite literals outside the `i64/i64` rational range are retained as f64;
  malformed and nonfinite literals are rejected with a parser error rather
  than being converted to zero.

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

## Historical Evaluator Evidence Consolidated From TODO (2026-10-22)

The following records were previously embedded in `View/TODO.md`. They are
preserved here as historical evidence; none authorizes a frontend default
change or a general Atom-vs-Expr performance claim.

### Combustion-like small callback fixture

Source report:
`test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story.md`.
Recorded at 2026-09-23T10:37:43.937Z; 3 unknowns, 7/9 Jacobian nonzeros,
20,000 callback evaluations per batch, telemetry off, no AOT.

| Matrix | Route | Prepare ms | Residual ns/call | Jacobian ns/call |
| --- | --- | ---: | ---: | ---: |
| Sparse | historical AtomView | 0.944 | 153.130 | 402.175 |
| Sparse | ExprLegacy | 0.169 | 134.955 | 276.005 |
| Sparse | AtomViewExprCompat | 0.217 | 135.250 | 405.610 |
| Banded | historical AtomView | 0.153 | 161.280 | 407.300 |
| Banded | ExprLegacy | 0.111 | 172.695 | 277.895 |
| Banded | AtomViewExprCompat | 0.172 | 143.085 | 419.635 |

Both Atom-derived routes had 151 Expr nodes, depth 9 and serialized length 361;
ExprLegacy had 127 nodes, depth 7 and length 310. Residual drift was zero and
maximum Jacobian drift was `7.105e-14`. This small fixture is not evidence for
large BVP behavior.

### Combustion-3000 callback distribution

The 2026-09-23 release stage-attribution story used 21 callbacks and measured
`2.823738 ms` ExprLegacy versus `3.336019 ms` AtomView, an AtomView gap of
18.47%. The earlier callback baseline was `3.060700 ms` versus `3.587467 ms`.
Sparse and Banded results were route-specific: Sparse measured
`1.228 +/- 0.264 ms` AtomView versus `4.006 +/- 0.209 ms` ExprLegacy over seven
samples, while Banded measured `9.763 +/- 0.307 ms` versus `8.268 +/- 0.284 ms`
respectively. These are separate layout findings, not a universal speedup.

### Real expression lowering and structural captures

The companion 2026-09-23 release story excluded matrix assembly:

| Workload | Route | Symbolic ms | Atom conversion ms | Closure ms | Eval ns/call | Max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| three-body | ExprLegacy | 2.653 | 0.000 | 0.045 | 4271.650 | 0.000e0 |
| three-body | AtomViewExprCompat | 0.000 | 0.082 | 0.080 | 4518.850 | 2.771e-12 |
| three-body | AtomNative | 0.000 | 0.490 | 0.254 | 2770.400 | 2.771e-12 |
| diffusion-chain | ExprLegacy | 1.057 | 0.000 | 0.027 | 488.950 | 0.000e0 |
| diffusion-chain | AtomViewExprCompat | 0.000 | 0.154 | 0.023 | 487.150 | 0.000e0 |
| diffusion-chain | AtomNative | 0.000 | 0.168 | 6.320 | 1928.150 | 0.000e0 |

The real LSODE2 structural report found 42 nonzeros for three-body: historical
AtomView and Compat each had 1,570 nodes, 207 unique fingerprints and depth 11;
ExprLegacy had 1,906 nodes, 244 unique fingerprints and depth 13. On the
128-variable diffusion chain all routes had 382 nodes, 14 unique fingerprints
and depth 1. Tree size alone did not explain the evaluator gap.

A subsequent debug decomposition measured three-body scalar evaluation at
`5109.4 ns/call` for Compat and `6179.9 ns/call` for ExprLegacy; diffusion-chain
measured `2707.25` and `3342.05 ns/call`, respectively. Preparation cost also
varied independently. These debug figures localize costs but are not release
performance evidence.

## Typed Policy, Prepared Scratch And AOT Contract Pass (2026-10-22)

This Debug-only pass closes the correctness/test work for the current View
contracts. It does not promote `PreparedAtomSystem` into solver hot paths and
does not make a performance claim.

- Added a domain/IEEE matrix across legacy Expr/Atom compatibility evaluation
  and explicit typed evaluation. It covers `log` at negative/zero inputs,
  `asin`/`acos` outside the real domain, cotangent singular/non-finite inputs,
  Atom-only `sqrt`, and overflow/non-finite propagation. Typed `cot(+inf)` is
  now classified as a domain error; legacy behavior remains IEEE-compatible.
- Added safe-constructor typed-error checks for malformed arity, unsupported
  functions, non-finite constants, callback/context failures and shape errors.
  Fallible Atom conversion now has tests for unsupported safe shapes. Internal
  defensive errors for malformed packed bytes are not tested by violating
  unsafe representation invariants.
- Extended the concurrent prepared-system test to verify each worker's
  scratch capacity remains stable over 64 numeric rebinds and that equivalent
  worker workloads reach the same high-water proxy. This observes retained
  container capacity only, not allocations or process memory.
- Strengthened AOT manifest tests: ExprLegacy and AtomNative have distinct
  artifact identity while preserving matching function ABI; ordered input ABI
  changes the key. Existing AOT lifecycle/registry tests validate readiness,
  stale/partial state, provenance handoff and cleanup. BVP_Damp Debug parity
  tests compile Atom AOT and compare it with Lambdify on Sparse and Banded.
- Added `benches/symbolic_prepared_memory_audit.rs`, a separate counting
  allocator audit for warmed sparse-chain residual/Jacobian calls at dimensions
  128 and 512. Both targets passed `cargo check`; the initial Debug pass did
  not collect Release allocation or timing results. Those are recorded in the
  later Release section below.

Debug verification on Windows, all with `--no-default-features`:

```text
cargo test --lib --no-default-features symbolic::View:: -- --nocapture --test-threads=1
194 passed, 0 failed, 8 ignored

cargo test --lib --no-default-features symbolic::View::conversions::tests -- --nocapture --test-threads=1
33 passed, 0 failed

cargo test --lib --no-default-features symbolic::View::prepared::tests -- --nocapture --test-threads=1
9 passed, 0 failed

cargo test --lib --no-default-features symbolic::codegen::codegen_manifest::tests -- --nocapture --test-threads=1
7 passed, 0 failed

cargo test --lib --no-default-features symbolic::codegen::codegen_aot_lifecycle::tests -- --nocapture --test-threads=1
3 passed, 0 failed

cargo test --lib --no-default-features symbolic::codegen::codegen_aot_registry::tests -- --nocapture --test-threads=1
8 passed, 0 failed

cargo test --lib --no-default-features numerical::BVP_Damp::test_parity_corpus -- --nocapture --test-threads=1
7 passed, 0 failed

cargo test --lib --no-default-features combustion_fixture_componentwise_callbacks_match_frontends_and_matrix_layouts -- --nocapture --test-threads=1
1 passed, 0 failed

cargo test --lib --no-default-features lsode2_combustion_view_componentwise_parity_across_states_and_parameters -- --nocapture --test-threads=1
1 passed, 0 failed

cargo test --lib --no-default-features combustion_bvp_componentwise_frontend_parity_pins_compiler_input -- --nocapture --test-threads=1
1 passed, 0 failed

cargo test --lib --no-default-features combustion_1000_accepted_trajectory_matches_frontends_and_layouts -- --nocapture --test-threads=1
1 passed, 0 failed

cargo check --bench symbolic_prepared_memory_audit --bench symbolic_prepared_runtime --no-default-features
passed
```

At the time of this Debug pass, remaining gates were Release allocation/timing
and binary/compile-time impact measurements, followed by a separately approved
solver integration decision. The allocation and callback measurements were
subsequently completed as recorded below. Debug timings and
allocator-instrumented timings must not be treated as normal runtime
performance.

The structural-counter release slice used 21 callbacks and measured
`2.823738 ms` ExprLegacy versus `3.336019 ms` AtomView. Its shape was 62,988
evaluators / 242,928 nodes. A separate cold console capture was approximately
`721.823 ms` AtomView versus `162.328 ms` ExprLegacy for generation, with
symbolic-Jacobian preparation and callback compilation as visible contributors.
Cold parent/child scopes must not be added together; use the dated typed report
for stage-level interpretation.

### Operation corpus conclusion

The 22-form corpus covers arithmetic shape, powers, functions, coefficients,
parameter lookup, n-ary trees and repeated subexpressions. Its structural
results were mixed: some forms shortened or improved, while explicit division
lowered to negative powers and function-heavy forms could grow or slow down.
The nested-power simplification is domain-sensitive. No global Atom tree
rewrite follows from this corpus; real LSODE2 and BVP callback parity and
release measurements remain mandatory.

## Prepared Runtime Release Audit (2026-10-22)

The Release memory audit and callback Criterion comparison ran on the Windows
development machine with `--no-default-features`. The allocation audit used
5,000 warmed evaluations per dimension; Criterion used its default 3-second
warmup and 100 samples. Preparation and solver/linear-algebra work are outside
the Criterion timed loop. Each sample evaluates the same sparse-chain
residual/Jacobian batch into preallocated output buffers.

### Allocation And Retained Scratch

| Dimension | Jacobian nnz | Allocations/eval | Allocated bytes/eval | Deallocations/eval | Deallocated bytes/eval | Scratch capacity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 255 | 0 | 0 | 0 | 0 | 8 |
| 512 | 1023 | 0 | 0 | 0 | 0 | 8 |

These are counting-allocator results for the warmed numeric callback path with
no custom functions; input/output and scratch storage are prepared before
counting. Audit-loop wall time was 14.791 ms and 59.032 ms respectively for
5,000 evaluations, but allocator instrumentation makes those values unsuitable
as ordinary callback timing. The capacity field is a proxy for the retained
scratch containers, not total memory or RSS. Per-worker stable capacity is
verified separately by the Debug concurrent rebind test.

### Callback Throughput

Criterion estimates (microseconds per full residual/Jacobian batch; brackets
are the reported estimate interval):

| Dimension | Expr Lambdify closures | Atom Lambdify closures | PreparedAtomSystem with caller scratch |
| ---: | ---: | ---: | ---: |
| 128 | 2.2229–2.2512 (2.2364) | 3.4057–3.4101 (3.4079) | 2.9394–2.9567 (2.9461) |
| 512 | 8.2481–8.2732 (8.2593) | 13.943–13.986 (13.964) | 11.716–11.737 (11.726) |

For these sparse-chain callbacks, `PreparedAtomSystem` was about 14% faster
than Atom Lambdify closures at dimension 128 and 16% faster at 512, but about
32% slower than Expr closures at 128 and 42% slower at 512. This is a
workload-specific API-level comparison, not a claim about all Atom plans or
solver performance. It gives no reason to replace Expr or change solver
defaults. Further adoption requires paired large solver callback and full
solve/continuation evidence, plus binary-size and compile-time impact.

The first Criterion attempt exposed a benchmark-fixture issue: manually built
symbols did not match interned IDs in converted Atoms in the optimized build.
The harness now constructs one named-input `PreparedAtomSystem`; the final run
passed its Expr/Atom/prepared numerical preflight at both dimensions before
collecting samples.

## BVP_Damp Combustion-3000 Prepared Runtime A/B (2026-10-22)

This addressed probe uses the production BVP_Damp combustion equations and
AtomView-discretized residuals, the Banded layout `(kl, ku) = (8, 3)`, sequential
execution, and telemetry Off. It compares the current solver-native AtomView /
Banded Lambdify callbacks with `PreparedAtomSystem` compiled directly from the
same retained Atom residuals and input order. The Prepared route writes into
caller-owned residual/Jacobian buffers and scatters into caller-owned compact
`BandedAssembly`; no dense matrix is materialized.

Correctness preflight on the 18,000-unknown, 62,988-entry problem:

| Quantity | Maximum absolute difference |
| --- | ---: |
| residual | 0 |
| compact banded Jacobian slots | 0 |

### Release latency

Windows development machine, optimized Cargo bench profile, allocator audit
disabled. The harness alternated route order across 21 paired samples, with
three callbacks per sample; table values are median/MAD per callback in ms.
Preparation is measured separately from warm callbacks. The Prepared setup
figure is the standalone runtime-preparation cost from the production Atom
residuals, not a complete replacement solver setup.

| Route | Setup ms | Warm callback median/MAD ms |
| --- | ---: | ---: |
| Production AtomView/Banded | 225.354 | 6.356 / 0.374 |
| PreparedAtomSystem + reusable band scatter | 74.021 | 3.180 / 0.179 |

The Prepared callback batch is about 50% shorter on this workload. This is
meaningful callback-level evidence, but it does not measure a complete Newton
trajectory, factorization, or continuation and does not establish end-to-end
solver speedup. The callback interfaces also differ in output ownership:
production returns newly assembled solver objects, while the Prepared row
reuses caller-owned buffers and compact storage. A solver adapter would need to
preserve safe ownership and concurrency while retaining that reuse benefit.

### Separate Debug allocator audit

The counting allocator was enabled only in a separate Debug-profile process;
its timings are deliberately excluded from the latency result. Allocator-live
bytes are not process RSS. Setup figures cover the measured allocations for
the production bundle and the additional Prepared runtime respectively.

| Phase | Allocations | Allocated bytes | Retained delta | Peak live delta |
| --- | ---: | ---: | ---: | ---: |
| Production AtomView/Banded setup | 5,198,536 | 398,171,680 | 79,611,557 | 122,836,349 |
| Prepared runtime preparation | 951,541 | 202,022,609 | 49,381,560 | 68,943,968 |
| Production warm callback | 32 / callback | 2,252,792 / callback | n/a | n/a |
| Prepared warm callback with reused outputs | 0 / callback | 0 / callback | n/a | n/a |

The zero-allocation Prepared figure applies only to the warmed numeric evaluator
and reusable caller-owned buffers for this built-in-function workload. It is
not a claim that arbitrary custom callbacks or a solver adapter are
allocation-free. The measured retained delta is substantial (about 49 MB), so
memory cost must remain part of any adoption decision.

### Decision

The hypothesis is supported for warm callback evaluation on this one large
BVP_Damp workload: parity is exact, callback latency improves, and reusable
outputs eliminate callback allocations. This is not yet proof that integrating
the runtime improves a full solver solve. Preparation cost, retained memory,
and the solver's output-ownership contract still need to be amortized against
real Newton callback counts. Keep `PreparedAtomSystem` as a low-level opt-in
capability; do not change BVP_Damp defaults or integrate it into production
solver callbacks on this evidence alone.

### Combustion-3000 Expr callbacks and full-solver gate (2026-10-22)

The callback harness was extended with the ExprLegacy/Banded closure route and
with an experimental end-to-end `NRBVP` run that installs the Prepared callbacks
through the explicit prepared-callback lifecycle. This does not change the
production default. The Release callback run used 21 interleaved samples of
three callbacks each, at 18,000 unknowns and 62,988 compact Jacobian entries;
execution was Sequential and callback telemetry was Off.

| Route | Setup ms | Warm residual+Jacobian callback median/MAD ms |
| --- | ---: | ---: |
| AtomView/Banded Lambdify closures | 203.309 | 6.302 / 0.288 |
| ExprLegacy/Banded Lambdify closures | 778.855 | 4.042 / 0.135 |
| PreparedAtomSystem + reusable band scatter | 74.240 | 3.173 / 0.051 |

On this callback-only measurement, Prepared was about `1.99x` faster than the
AtomView closure and `1.27x` faster than the ExprLegacy closure. ExprLegacy was
also faster than the AtomView callback in this particular run. These callback
times are not full solver speedups: the Prepared row reuses caller-owned output
buffers while the closure rows return newly assembled solver objects, and its
setup/retained-memory cost still matters.

The original full-solve gate in `benches/bvp_damp_prepared_atom_ab.rs` was not
an independent-solve parity test. It reused mutable solver objects across
samples, read cumulative counters, and stopped on the Prepared route's second
same-object invocation before running the matching second AtomNative call.
Thus the reported `10` Prepared iterations / `21` residual requests combined
two invocations; they were not the counters of one solve. Its `20.993 ms`
sample is not valid for a matched full-solve speed comparison.

**Decision:** retain PreparedAtomSystem as a low-level opt-in route and make no
full-solver speed claim. The harness has been changed to prepare fresh solver
objects for each timed sample, with one solve per object, per-sample numerical
parity, and per-solve counters. Same-object replay is a distinct lifecycle
contract; the old harness did not compare its result symmetrically and cannot
answer whether such replay is supported for this large fixture.

### Debug combustion trajectory diagnostic (2026-10-22)

Added `prepared_bvp_diagnostics::combustion_prepared_route_preserves_debug_iteration_trace`
under the `symbolic::View` test module. This is a correctness test, not a
benchmark. It compares solver-native AtomView/Banded with the experimental
PreparedAtomSystem/Banded adapter on the same combustion fixture and records
every residual/Jacobian callback input state.

At 3,000 mesh steps in Debug, with three independent fresh solver pairs (one
solve per solver object):

- Each of the three native/Prepared pairs converged in 5 iterations, with 12 residual calls,
  1 Jacobian call, 5 damping trials and 10 linear solves; final state difference
  was exactly zero. Per-solve counters matched exactly between routes.
- Initial and converged-state residuals and compact Jacobian slots matched
  exactly. Ordered callback input/output traces matched at every event.

The earlier same-object replay had not compared all routes or separated
cumulative counters. A dedicated Debug replay test now covers AtomNative,
PreparedAtom and ExprLegacy symmetrically at 3,000 steps. The first solve is
identical for all routes. A raw second call, without resetting the mutable
solver state, returns no converged solution for all three; each performs 5
iterations, 9 residual requests, 2 Jacobian requests, 3 damping trials and 8
linear solves. This is a replay from the solver's current modified iterate, not
an independent solve from the original guess.

Calling `try_set_initial_guess` with the original 6-by-3000 guess before the
next call restores the independent-solve trajectory while preserving the
prepared callbacks. All three routes then converge again in 5 iterations with
the original per-solve work (12 residual requests, 1 Jacobian request, 5
damping trials and 10 linear solves), and their final states match. Numerical
telemetry counters are cumulative by design; per-call results must be obtained
by subtracting snapshots. The independent-sample Release benchmark remains
the speed comparison and now includes all three routes; it does not time raw
same-state replay as if it were a fresh solve.
