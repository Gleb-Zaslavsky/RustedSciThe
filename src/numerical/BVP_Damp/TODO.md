# BVP_Damp Architecture Review And Migration TODO

Review date: 2026-09-19. Status: architecture review plus an intermediate test
suite migration. The numerical suite is not in its final physical layout yet;
existing tests remain intact while descriptive module aliases and shared test
helpers are introduced. New release numbers must not be treated as final until
the post-migration corpus is rerun.

This is the BVP_Damp execution plan for the shared
[symbolic migration](../../symbolic/TODO.md). BVP Damped/Frozen is the first
solver family to migrate; LSODE2 follows after the shared components are proven
here. Reuse the [story registry index](BVP_DAMP_STORY_TESTS.md) and its
thematic ledgers: [correctness](BVP_DAMP_STORY_CORRECTNESS.md),
[symbolic](BVP_DAMP_STORY_SYMBOLIC.md), [AOT](BVP_DAMP_STORY_AOT.md), and
[heavy performance](BVP_DAMP_STORY_PERFORMANCE.md).
Earlier review opinions in [src/TODO.md](../../TODO.md) are historical input,
not proof that a reported defect still exists.

## Optimization Evidence Policy

Every performance-sensitive refactor follows the same evidence order:

- instrument the changed preparation/hot-path/linear-runtime boundary first;
- perform several debug correctness and diagnostic passes before spending time
  on a release build;
- only then run the dated release story and compare against the previous
  baseline, with telemetry identifying any regression stage;
- keep unresolved release comparisons explicitly provisional and do not call a
  backend production-ready while correctness, invalidation, or stage parity is
  open.

Telemetry schemas and typed error propagation are part of the production code
change, not follow-up polish.

## 0. Test Suite Architecture: Intermediate Migration State

This section deliberately describes the current transition, not a finished
test-suite redesign. The priority is to prevent a test from disappearing while
ExprLegacy and AtomView are being compared. The old ExprLegacy cases remain a
regression oracle and are not to be deleted or silently rewritten.

- [x] Inventory the existing test files into correctness, classic examples,
  backend comparison, AOT diagnostics/race stress, and factorization-cache
  lanes.
- [x] Give the `BVP_Damp.rs` registrations descriptive internal module aliases
  (`test_correctness`, `test_classic_examples`, `test_backend_compare`,
  `test_aot_diagnostics`, `test_aot_race_stress`, and
  `test_factorization_cache`) without physically moving files yet.
- [x] Update self-spawned child-test paths after the alias change so isolated
  AOT stories still execute the intended test.
- [x] Add `tests/common.rs` as the first shared seam for route labels, common
  initial guesses, repetition parsing, and componentwise solution comparison.
- [x] Add an explicit architecture lane comment to each current test module;
  this is documentation of the intermediate state, not a claim of parity.
- [ ] Classify every existing case as `ExprLegacy oracle`, `AtomView parity`,
  `backend-independent`, or `performance/AOT`, and record the classification in
  the story registry with a date and command.
- [ ] Extend `common.rs` with shared BVP fixtures and typed assertions only
  after the first classification pass shows genuine duplication. Do not make a
  broad helper abstraction merely for symmetry.
- [x] Add a dedicated parity corpus that runs the same problem, initial guess,
  tolerances, and boundary conditions through ExprLegacy and AtomView. Keep
  correctness assertions separate from wall-clock assertions. The first
  deterministic Banded Lambdify corpus is now in
  `tests/parity_corpus.rs` (linear two-point and oscillator fixtures).
- [ ] Add the singular/endpoint and pure-numeric AtomView cases from the
  existing `tests/` corpus; these are correctness gates before performance
  claims.
- [ ] Rerun debug correctness first, then release story tests after the
  no-Mutex Jacobian and Atom-native assembly work stabilizes. Mark old numbers
  as historical rather than overwriting them.
- [ ] Only after the corpus is classified and green decide whether physical
  file splitting or final renaming improves navigation. Until then, avoid a
  large rename-only diff and preserve the current paths.

Migration invariant: every legacy test must remain runnable through a stable
qualified path, and every new AtomView result must identify its route explicitly
in output and documentation.

## 1. Findings And Priorities

Confirmed means visible in the inspected call chain. It does not mean that the
size of a speedup is known. Proposed changes require isolated measurements.

| Priority | Finding and code evidence | Consequence / decision |
| --- | --- | --- |
| P1 | Damped `step` calls `MatrixType::solve_sys` for the initial step and damping candidates; dense LU, faer `sp_lu`, and banded `build_solver_for_system` factor on each call. | A cached Jacobian is not a cached factorization. Separate factorization from repeated RHS solves; quantify this first. |
| P1 | Frozen `iteration` clones `old_jac` when reusing it, then calls the same factor-and-solve API. | Frozen avoids symbolic/numeric Jacobian evaluation, but still copies the matrix and repeats factorization. Borrow the cached matrix and retain its factorization. |
| P1 | BVP dense Lambdify locks the output matrix per nonzero; faer sparse Lambdify locks a triplet vector and reconstructs CSC each evaluation. See `symbolic_functions_BVP.rs`. | Prepare destination structure once and fill disjoint numeric buffers. Preserve runtime threshold semantics explicitly. |
| P1 | `numeric_discretization::build_numeric_generated_solver_state` analytical Jacobian callback allocates `n_unknowns^2` entries and scans all of them before `from_vector`. `finite_difference_jacobian` also allocates dense storage plus quadratic triplet capacity. | Pure numerical Sparse is not sparse during assembly. Direct stencil assembly and structured FD deserve their own memory/performance gate. |
| P1 | `VectorType` arithmetic returns new `Box<dyn VectorType>`; `to_DVectorType` owns a copy. Sparse subtraction and banded callback/solve adapters perform conversions. | Buffer ownership and borrowed views are stronger optimization candidates than the virtual call alone. |
| P1 | BVP Atom assembly materializes residuals and nonzero derivatives back into Expr in `install_atom_discretized_system` and `calc_atomview_sparse_jacobian_with_bandwidth`. | Retain the packed representation through evaluation; measure preparation separately from callback execution. |
| P1 | Runtime `step`, matrix solves and callback wrappers contain panic/expect paths below public `try_*` entry points. | Typed setup errors do not yet guarantee typed numerical failures. Design a fallible callback/linear-solve boundary. |
| P1 | Statistics use string maps; `linear_system` time includes factorization; `checkmem` returns dense-equivalent MiB, cast to integer under `jacobian memory, MB`. | Add typed raw telemetry, separate factor/solve durations and actual storage estimates before making architectural claims. |
| P2 | Public solver fields, legacy method strings, matrix overrides and generated-backend options coexist. | Resolve one validated runtime plan at preparation boundaries and define cache invalidation. |
| P2 | `Fun`, `Jac`, `VectorType`, `MatrixType`, partial enums and `Any` downcasts overlap. | Use a closed internal matrix/backend representation; preserve public extension and compatibility adapters. |
| P2 | Damped and Frozen duplicate setup/reporting but have different iteration and mesh contracts. | Share preparation and linear runtime; retain separate numerical policies and compatibility semantics. |

### Corrections To Broad Migration Assumptions

- Banded Lambdify already has a mutex-free diagonal write path in
  `symbolic_functions_BVP2::generate_banded_jacobian_assembly_from_plan_parallel`.
  Its entry-chunk alternative evaluates a temporary list before scattering.
  Both currently allocate a new `BandedAssembly`; neither should be described
  as the dense/sparse mutex implementation.
- The Nonlinear systems prepared no-Mutex implementation still compiles Expr
  scalar callbacks in `symbolic_legacy.rs`. Reuse its buffer ownership and
  partitioning ideas; it is not evidence that an Atom evaluator is faster.
- `somelinalg/banded/linear_solver.rs` already provides a `LinearSolver` enum
  implementing `DirectLinearSolver::solve_in_place`. Extend or compose this
  working infrastructure instead of introducing another banded LU algorithm.
- The production backend set is deliberately closed to nalgebra `Dense`, faer
  `Sparse`, and the local native `Banded` solver. Sparse storage also contains
  the historical sprs `Sparse_1` experiment; keep it callable through an
  explicit compatibility route, but do not spend production performance or
  feature-parity budget on it unless a concrete downstream user requires it.
- Current matrices and vectors are public extension points (`pub fun`, `jac`,
  and `y`, plus public traits). Replacing their types directly would break users.

## 2. Decision: Closed Internal Backends, Open Callback Boundary

Proposed internal design: an explicit backend enum for built-in matrix/linear
runtime selection, dense contiguous Newton state and work vectors, and narrow
borrowed/in-place operations. Keep dynamic callbacks for user closures and
linked/generated evaluators where runtime polymorphism is appropriate.

| Choice | Assessment |
| --- | --- |
| Existing boxed matrix/vector traits throughout arithmetic | Compatible, but permits mismatched pairs, runtime downcasts, allocation-returning arithmetic and implicit conversion. Retain as reference/adapter path. |
| Internal enum for built-in backend runtimes | Preferred: explicit variants, exhaustive dispatch, native matrix/factor ownership and no `Any` in built-in arithmetic. A match is not automatically faster than a vtable call. |
| Generic nonlinear loop over a backend trait | Possible later, if benchmarks justify monomorphization cost and complexity. CLI runtime selection still requires a dispatch boundary. |
| Remove every trait object including closures | Rejected: callback implementations remain open-ended, and this does not address matrix copying or factorization lifetime. |

- [ ] Prototype internal `BvpLinearRuntime` with Dense, FaerSparse and
  NativeBanded as the only first-class variants. Keep SprsLegacy/other matrix
  implementations outside that production runtime behind compatibility
  adapters. Exact names are provisional.
- [ ] Keep storage choice distinct from solver policy: banded matrices can use
  different native algorithms, and sparse direct/iterative solves have different
  caches. Do not equate every Sparse choice with LU.
- [ ] Preserve the existing native banded solver enum and pivot/refinement
  policy. Box a large variant if useful for enum size; that is still closed
  dispatch and not a per-operation trait-object allocation.
- [ ] Prefer dense state/residual/step buffers even for sparse Jacobians.
  Verify borrowed nalgebra/faer view interoperability before choosing the
  owning buffer type. Do not assume faer buffers are always slice-compatible.
- [ ] Add slice/view-based norm, trial update and copy-into operations. Compile
  closed dispatch once per operation, not inside per-element loops.
- [ ] Keep the current public trait signatures through compatibility facades.
  Explicit custom callbacks use a compatibility route; never silently bypass
  user-installed `fun`/`jac` with cached generated callbacks.
- [ ] Benchmark dispatch-only differences with identical ownership, then
  separately benchmark ownership improvements. Do not attribute both to enums.

## 3. Proposed Ownership And Invalidation Model

```text
Problem + Options -> validated ResolvedBvpPlan
    -> PreparedMeshProblem (mesh/BC/order/stencil and callback preparation)
    -> BoundProblem (validated parameter snapshot)
    -> SolveWorkspace (state, residuals, candidate steps, callback scratch)
    -> JacobianCache (numeric values + generation)
    -> LinearRuntime (analysis + factors for that generation)
    -> SolveReport / postprocessing dataset
```

Symbolic preprocessing, Lambdify/AOT execution and matrix storage are separate
choices. The same linear runtime must also support pure numerical callbacks.
The shared symbolic layer stays under `symbolic`; BVP mesh and boundary
elimination stay in BVP code. Damped/Frozen share infrastructure, not one forced
nonlinear algorithm.

- [x] Define the shared prepared mesh/layout, parameter-binding, numeric
  Jacobian and factor lifecycle states (2026-09-20). `BvpRuntimeRevision` now
  distinguishes `Prepared`, `NumericalJacobianCurrent`, `FactorCurrent` and
  `Invalidated`; dropping a factor preserves a valid prepared callback/Jacobian
  plan, while a structural revision invalidates the complete plan.
- [ ] Turn the metadata guard into the complete common `PreparedPlan` owner of
  mesh/layout, callbacks, Jacobian generation and factor generation. The first
  resource-owning slice is now implemented and debug-tested (2026-09-20):
  Damped and Frozen store their reusable Dense/faer factor and numeric
  Jacobian in the shared `BvpPreparedRuntime` container. Live callbacks,
  mesh/layout and Banded-native ownership remain compatibility fields until
  their own migrations are complete; this is deliberately not hidden behind
  the stage enum.
- [ ] Reuse factors while the algorithm intentionally reuses the same numeric
  Jacobian, even when a damping candidate changes state or residual.
- [ ] Invalidate factors when numeric Jacobian values are replaced. Invalidate
  structural analysis on mesh/BC layout/order/pattern changes; invalidate solver
  state when linear policy changes.
- [x] Make Damped `set_p`, `set_new_step`, `set_params`, and
  `set_param_values` invalidate cached numeric Jacobian/factor state. Debug
  gates cover continuation and parameter rebinding; shared plan-level
  invalidation and generated-callback rebinding remain open.
- [x] Introduce the shared allocation-free `BvpRuntimeRevision` stamp
  (2026-09-20). Damped and Frozen now track mesh, parameter binding, callback,
  and solver-configuration generations through the same type; a prepared
  runtime is current only when all captured generations match. The shared
  prepared factor owner and public revision diagnostics remain open.
- [x] Add a separate problem-generation lane to the shared revision stamp
  (2026-09-20). Damped and Frozen now expose revision-tracked
  `set_boundary_conditions` setters which invalidate factors, callbacks and
  published results before a prepared solve can be reused. Direct writes to
  historical public fields remain compatibility-only and are intentionally not
  claimed to be safe; an encapsulated `PreparedPlan` boundary is still open.
- [x] Parameter-only binding reuses symbolic preparation and invalidates the
  numeric factor (2026-09-20). The Dense/faer/Banded solver-surface test now
  asserts both a factorization invalidation on rebind and a new factorization
  during the subsequent prepared solve.
- [ ] Preserve current parameter specialization: Banded Lambdify may substitute
  values during compilation. A runtime-rebind mode must not reuse a specialized
  evaluator with different values. Make specialization versus runtime binding
  explicit and benchmark both.
- [x] Validate/snapshot public mutable configuration at the prepared-solve
  boundary (2026-09-20). The fingerprint guard rejects direct compatibility
  field mutation without adding hashing to residual/Jacobian hot paths.
  New encapsulated state may guarantee stronger invariants; legacy field writes
  must not silently leave stale caches. Avoid structural hashing per callback.
- [ ] Preserve Damped interval-count and Frozen point-count mesh conventions
  through adapters (`solver_common.rs` already tests these differences).

## 4. Linear Factorization And Buffer Work

- [ ] Instrument factorization count/time separately from RHS solve count/time
  in the existing path to establish a baseline. Native Banded is now covered;
  factorization ownership/reuse for Dense/faer still requires dedicated
  adapters.
- [x] Add the first typed no-Mutex factor/RHS duration baseline to native
  `BandedMatrixType`; invalidation resets both duration scopes together with
  the cached factor.
- [x] Add compatibility-safe `MatrixType::solve_sys_with_timing`. Dense, faer
  direct LU, and native Banded now expose typed factorization/RHS durations;
  the legacy `solve_sys` contract remains unchanged and unknown external
  implementations use an explicitly coarse fallback classification.
- [x] Add the additive `MatrixType::try_solve_sys` and
  `try_solve_sys_with_timing` boundary (2026-09-19). Typed Damped steps now
  use it instead of the bare legacy solve call. Dense distinguishes dimension
  mismatch, unsupported policy, non-finite/singular output and factorization
  failure; faer maps direct/iterative failures; native Banded preserves the
  factory and RHS `Result` values. `CsMat` and external implementations remain
  behind the explicit compatibility adapter until their native fallible APIs
  are migrated.
- [x] Extract the typed Dense/faer/Banded linear-boundary correctness gates
  into `tests/linear_solve_boundary.rs` (2026-09-20). Keep historical
  `BVP_traits` adapter tests separate so production-boundary and compatibility
  evidence remain independently runnable.
- [ ] Optional compatibility work: extend the typed linear boundary to `CsMat`
  only if a downstream caller needs it. This is not a production-backend
  milestone; add merely a sprs-specific no-panic/correctness gate if we touch
  the route.
- [ ] Add solver-level parity gates for typed dimension/factorization errors
  across Dense, faer and Banded, including partial telemetry preservation.
- [x] Add the first Damped solver-level typed-error/partial-telemetry gate
  (2026-09-20). Shape mismatch and singular Dense factorization now preserve
  residual telemetry and avoid false linear/factorization counters; the
  backend-specific faer/Banded matrix gates remain separate evidence.
- [x] Move the owned Dense/faer factor workspace from two solver-local option
  fields into the shared `BvpPreparedRuntime` container (2026-09-20, debug and
  release correctness gates). Damped and Frozen preserve the same factor-count
  and invalidation behavior; the common container has a direct
  ownership/invalidation unit gate.
- [ ] Complete the ownership migration for callbacks, mesh/layout and
  Banded-native factor state, then add factor-count and
  cache-invalidation gates at the full-plan boundary. Typed timing alone is
  not evidence of complete plan ownership.
- [x] Harden owned Dense/DenseBanded/faer factor solves against non-finite
  results and DenseBanded panic leakage (2026-09-20). Singular large Dense
  factors now surface a typed `LinearFactorError` instead of publishing a
  non-finite Newton step; the release performance impact remains to be measured.
- [x] Add an internal, non-legacy Dense/faer factor-owner prototype to Frozen;
  debug parity, backward-error, invalidation, and the dated release story all
  pass. This is still not a production promotion because shared-runtime
  generalization is open.
- [x] Extend the same non-legacy factor-owner prototype to Damped Dense/faer
  solves. The first RHS reports factorization time and repeated RHS solves do
  not refactor; the debug parity gate is green. This remains provisional until
  both strategies use one prepared runtime and release data is rerun.
- [x] Unify the Damped and Frozen Dense/faer prototypes on the shared typed
  `OwnedLinearFactorRuntime` (2026-09-19). Both strategies now use the same
  `try_solve` factor/RHS timing and factor-cache-hit semantics; Frozen
  invalidation is covered by a dedicated correctness gate. This closes the
  solver-local duplication, but does not close shared plan-level invalidation
  or release performance validation.
- [x] Diagnose the anomalous first faer Frozen wall-clock sample: non-AOT
  Lambdify diagnostics were triggering the one-time Rayon Auto calibration.
  The calibration now runs only for a compiled AOT runtime, so Lambdify does
  not pay this unrelated initialization cost.
- [x] Re-run the dated release story after the calibration fix and replace the
  provisional faer wall-clock snapshot. The `466 ms` anomaly disappeared; keep
  the new table provisional until the shared factor-owner runtime is complete.
- [x] Add a dated provisional release benchmark/story for repeated Frozen
  Dense/faer/Banded solves. The scaffold is intentionally not a production
  baseline and must be rerun after factor-owner validation.
- [ ] Split prepare/analyze, numeric factor and `solve_into` operations. Cache
  factors once per numeric Jacobian generation with the existing pivot policy.
- [ ] Reuse existing `LinearSolver`/`DirectLinearSolver` for banded systems and
  existing faer/nalgebra factors; do not add a new LU implementation.
- [ ] Investigate reuse of faer symbolic analysis for fixed patterns separately
  from numeric factor reuse. Never reuse numeric factors after value changes.
- [x] Remove Frozen full-Jacobian clones on unchanged-Jacobian iterations.
  Frozen now borrows the cached Jacobian during reuse; the dedicated Sparse
  and Banded AtomView linear-solve gates remain correctness coverage.
- [ ] Add reusable current/candidate/residual/step buffers; preserve current
  damping trial order and acceptance decisions, including bounds checks.
- [x] Compact the AtomView direct Banded diagonal plan to compile and evaluate
  only structurally present slots (2026-09-20, debug correctness green). The
  freshly allocated assembly remains zero-initialized, so structural zeros no
  longer pay per-callback evaluator/branch/write cost; release speedup remains
  intentionally unclaimed until the accumulated baseline rerun.
- [x] Remove the unconditional dense-state clone at the Banded Lambdify
  callback boundary (2026-09-20, debug correctness green). Built-in
  `DVector`/`YEnum::Banded` states are now borrowed through a `Cow` view;
  unknown external `VectorType` implementations retain the compatibility
  copy fallback. Release impact remains part of the same pending pass.
- [x] Avoid flattened argument-buffer allocation for unparameterized AtomView
  Dense/Sparse residual and Jacobian callbacks (2026-09-20, debug correctness
  green). Parameterized callbacks retain the `[parameters..., unknowns...]`
  ABI; release impact remains part of the accumulated baseline rerun.
- [x] Avoid the temporary entry-value `Vec` in sequential AtomView Banded
  `EntryChunks` evaluation (2026-09-20, debug correctness green). Sequential
  callbacks now evaluate and scatter directly into their owned assembly;
  parallel callbacks retain collect-then-scatter to keep writes lock-free.
  Release impact remains unclaimed until the accumulated baseline rerun.
- [ ] Preserve iterative-method initial guesses and stopping tolerances; LU
  caching rules cannot be applied blindly to GMRES/preconditioner state.
- [ ] Require linear residual/backward-error checks and complete repeated-RHS
  parity across mesh, parameters, values and policy changes. Parameter and
  policy invalidation gates exist; fixed-CSC, band-slot and all-value parity
  still need a dedicated matrix-layout corpus.
- [x] Close the current solver-local factor-owner invalidation matrix (2026-09-20):
  Damped mesh/RHS/Jacobian setter changes and Frozen parameter-name/value changes
  now clear `old_jac`, owned factors, and the reuse window; a changed Damped mesh
  also updates `n_steps`, the initial guess shape, and the current state shape.
  An identical mesh is a no-op. Shared prepared-plan
  invalidation, value/policy coverage, and release validation remain open.
- [x] Add a native Banded cache-invalidation gate for solver-policy changes
  (2026-09-20). `set_solver_config` now has regression coverage proving that
  cached LU and stage timings are discarded before the next RHS solve; assembly
  replacement was already covered. Full cross-backend value/policy matrix and
  release validation remain open.
- [x] Reject stale Damped prepared callbacks after generated-backend policy
  changes (2026-09-20). Runtime configuration setters now pass through the
  central config transition, clear numeric factor state, and mark the prepared
  runtime dirty. `try_solver_prepared` returns typed
  `PreparedRuntimeInvalidated` until `try_eq_generate` rebuilds the callbacks.
  Frozen/shared-plan policy invalidation and release validation remain open.
- [ ] Compare nonlinear traces against the reference with the same evaluator.
  Iteration-count changes require explanation before accepting an optimization.
- [x] Add a first ExprLegacy/AtomView nonlinear trace parity gate (2026-09-20).
  The pure Lambdify Banded corpus now compares final state, iteration count,
  damping trials/rejections, refinement count, and the ordered typed decision
  event trace. The adaptive fixture now exercises one refinement and compares
  the mesh-revision event sequence. Its current Newton path has zero rejected
  trials; a reproducible rejected-trial fixture remains an explicit follow-up
  rather than a synthetic assertion.

## 5. Symbolic And Callback Migration

- [ ] Extract reusable input schemas, fixed nonzero layouts, caller-owned
  outputs and sequential/parallel policy below the solver layer.
- [ ] First allow existing Expr scalar evaluators to populate those buffers.
  This gives a controlled comparison for removing mutexes/allocations before
  changing arithmetic representation.
- [ ] Replace faer shared triplet insertion with deterministic prepared CSC
  slots. Populate values without sorting/reconstructing the pattern per call.
- [ ] Preserve already-working Banded diagonal partitioning. Add caller-owned
  assembly and an explicit Sequential/small-work fallback.
- [ ] Retain residual and derivative atoms from `DiscretizedBvpAtomSystem` and
  `PreparedSparseAtomSystem`. Compile direct Atom callbacks without compulsory
  Atom-to-Expr materialization; compatibility getters may materialize on demand.
- [ ] Validate direct Atom numerical semantics: coefficient conversion,
  supported functions, domains, reassociation, non-finite values and typed
  errors. Normalization can change rounding, so trace differences need numerical
  investigation rather than weakened assertions.
- [x] Add the first non-finite callback gate for the direct Atom/Banded
  Lambdify route. Residual and Jacobian callback values are validated before
  thresholding, so `NaN`/`Inf` cannot silently become a structural zero. The
  compatibility `Fun`/`Jac` ABI remains unchanged; solver-level propagation for
  non-banded compatibility callbacks is a separate follow-up.
- [x] Add a typed `BvpAtomDiscretizationError` boundary for invalid schemes
  in the Atom-native BVP assembly. `try_eq_step_atom` and
  `try_discretization_system_bvp_par_atom(_native)` now propagate errors while
  the historical constructors remain compatibility panic wrappers. This is a
  focused migration slice; full fallible callback propagation remains open.
- [ ] Keep preparation and runtime policy independently observable. Use the
  actual nonzero/expression workload to partition; report active workers/jobs
  and fallback rather than assuming a requested parallel mode ran in parallel.
- [x] Remove unreachable general-evaluator bookkeeping from the AtomView
  numeric-only callback path (2026-09-20). Prepared Atom evaluators now use a
  separate thread-local numeric tape path when the expression contains only
  constants, variables, arithmetic and builtins; custom-function evaluation
  keeps the general path. The release Banded baseline showed a provisional
  improvement for `combustion-3000` AtomView (`15.723 -> 14.271 ms` sampled
  evaluator, `1.570 -> 1.348 ms` solver evaluator delta) with unchanged
  componentwise parity. This is an optimization checkpoint, not final
  production ranking; the remaining tape/dispatch cost needs a larger corpus.
- [ ] Close the remaining AtomView scalar-evaluator gap before declaring the
  Banded route performance-ready: the 2026-09-20 release checkpoint is still
  about `1.89x` slower than ExprLegacy in the sampled evaluator and `2.72x`
  slower in the solver evaluator delta. Use a focused instruction-tape/
  dispatch benchmark and preserve the ExprLegacy ratio as the comparison gate.
- [ ] Preserve intentional numerical pruning separately from structural zeros.
  Current dense/sparse callbacks use `P.abs() > T`; banded has its own threshold.
  Test tiny derivatives, zero crossings and NaN/Inf explicitly before unifying
  policies. A non-finite value must not silently disappear as a zero.
- [ ] Feed existing AOT builders the prepared data through adapters. Keep
  artifact lifecycle/ABI changes outside the first callback migration.

- [x] Add fallible Damped/Frozen preparation validation (2026-09-20). Typed
  `try_task_check()` rejects malformed dimensions, interval/tolerances, scheme,
  matrix method, boundary/bounds data and Frozen strategy parameters without
  process exit; advisory sysinfo memory inspection is isolated from this
  boundary as well. Legacy `task_check()` and panic constructors remain only
  as compatibility wrappers. Direct symbolic callback panic boundaries remain
  a separate migration slice.

## 6. Pure Numerical Assembly

- [ ] Build a mesh stencil plan mapping local continuous derivatives directly
  to reduced dense/CSC/banded destinations after boundary elimination.
- [x] Assemble the numeric analytical Jacobian directly as structural
  triplets for Sparse/Banded consumers; Dense retains the legacy row-major
  staging for compatibility. This removes the unconditional `N^2` buffer and
  scan from the structured numeric path. A full mesh-stencil plan remains a
  separate follow-up optimization.
- [x] Hoist the numeric `full_to_unknown` map out of the Jacobian callback.
  Reusing node-to-reduced-column metadata is now part of the direct structured
  assembly path; fixed-value buffer reuse and a complete stencil plan remain
  follow-up work.
- [x] Reuse callback-local node-state buffers for `y0/y1` while preserving the
  public closures. The focused benchmark shows roughly 66% fewer allocations
  and 43-53% lower structured callback medians at 512 steps. Full residual,
  local-Jacobian and final-matrix workspace reuse remain follow-up work.
- [x] Establish the actual Banded Numerical route: `YEnum::Banded` now builds
  a compact scalar `BandedAssembly` from the numeric Jacobian triplets instead
  of silently selecting `DMatrix`. The solver-level gate
  `damped_banded_solver_reports_factorization_reuse_at_solver_level` proves
  the route with one factorization and repeated RHS solves; the existing
  numeric Banded correctness corpus remains the numerical parity gate.
- [ ] Add structured FD as an explicit follow-up: differentiate local RHS or
  use graph coloring only with a validated dependency pattern. Preserve the
  global FD implementation as the comparison oracle and fallback.
- [ ] Treat bound-aware perturbations, step-size changes and coloring as
  separate numerical changes with analytical-vs-FD parity and domain tests.
- [ ] Report full residual calls, local RHS calls and FD perturbations
  separately, plus dense staging bytes and actual matrix format.

## 7. Errors, Configuration And Common Infrastructure

- [x] Add the first fallible residual/Jacobian callback boundary. `Fun::try_call`,
  `Jac::try_call` and `try_finite_difference_jacobian` capture legacy callback
  panics and malformed FD output without changing the historical `call` ABI;
  Damped/Frozen `try_*` paths map them to `CallbackExecutionFailed` with the
  stage. Factor/solve context and iteration metadata remain follow-up work.
- [x] Add the first typed factor-owner `try_solve` boundary with explicit
  dimension-mismatch errors. The compatibility `solve` wrapper remains
  panic-based until Damped/Frozen expose a fallible iteration boundary.
- [x] Route Frozen `try_main_loop` through a typed `try_iteration` boundary;
  owned-factor failures now carry backend and matrix/RHS dimensions. The old
  `iteration` method remains a compatibility panic-wrapper.
- [x] Route Damped `try_main_loop_damped` through typed `try_damped_step` and
  `try_step_with_linear_telemetry` boundaries. Missing cached Jacobians,
  shape mismatches and non-finite residual/update values now reach the
  fallible loop; historical `step`/`damped_step` wrappers retain their old
  panic-on-error behavior.
- [x] Extend the Damped no-AOT `try_*` path across residual and Jacobian
  callback boundaries. `try_calc_residual` validates residual shape/finiteness
  without extra matrix conversion, and `try_recalc_jacobian` validates native
  matrix shape before factor preparation. Typed callback errors retain the
  stage and expected/actual dimensions; legacy methods remain wrappers.
- [x] Propagate the primary callback, singular-factor, dimension-mismatch and
  non-finite numerical failures through the typed Damped/Frozen `try_*` paths,
  retaining partial telemetry (2026-09-20). Compatibility panic wrappers and
  external MatrixType implementations remain explicitly outside this claim.
- [ ] Resolve method strings, matrix overrides, symbolic assembly, evaluator,
  AOT options, scheme and linear policy once into `ResolvedBvpPlan`.
- [ ] Use current typed task-parser/settings builders as input. Preserve
  documented aliases and reject incompatible combinations explicitly.
- [ ] Share prepared-state handoff and reporting for Damped/Frozen; preserve
  their distinct refresh and convergence policies.
- [ ] Keep adaptive-grid marking and construction unchanged during runtime
  work. Rebuild the prepared mesh problem transactionally after refinement and
  invalidate old factors/layouts. Verify from coarse 10-20 point starting grids.
- [ ] Preserve the AOT lifecycle lock in `generated_solver_handoff.rs`. It
  protects cold publication/load and is not the per-entry Jacobian mutex.
  Narrowing it requires dedicated concurrency evidence, outside this pass.
- [ ] Retain the existing postprocessing facade. Report/dataset adapters should
  consume finalized results without forcing matrix densification during solves.

## 8. Telemetry In Every Implementation Slice

- [ ] Treat the existing `tests/` diagnostic harness as the current evidence
  layer, not as a substitute for solver telemetry. It already provides useful
  repeated release runs, cold/warm separation, stage tables, aggregation and
  isolated child-process measurements. The runtime API must expose the same
  facts directly so callers do not need to reverse-engineer them from logs or
  reproduce private test helpers.
- [x] Add the first typed telemetry slice: `BvpTelemetryCounters` and
  `BvpTimingSnapshot` now carry raw counters/durations, while the old maps
  remain presentation adapters. Completion-frozen totals and solve-local
  decision/callback ownership are implemented; richer nested timing scopes
  remain follow-up work.
- [x] Move solver counter mutation behind the allocation-free
  `BvpTelemetryRecorder` (`Cell` fields). This permits residual accounting from
  `&self` callback boundaries without `RefCell`/`Mutex` borrow contention;
  `snapshot()` is the only conversion to the public immutable counters.
- [x] Add typed operation counters for scalar evaluations, residual/Jacobian
  requests, residual/Jacobian chunks, conversions and copies. The historical
  `residual_calls`/`jacobian_recalculations` fields remain unchanged and the
  new fields are additive, so old story tables keep their meaning.
- [x] Add the first typed scope hierarchy (`solve`, `mesh_revision`,
  `iteration`, `damping_trial`). Solve, mesh-refinement and iteration events
  are projected from existing runtime facts; damping-trial events and rejected
  trials are now recorded directly by the damped line-search loop. Timing of
  individual trials remains a separate follow-up instead of being guessed from
  total wall-clock time.
- [x] Add explicit storage-byte metadata with dense and compact-banded
  estimates. Estimates are marked as estimates; sparse/factor/scratch values
  remain unavailable until the native owners expose their real storage.
- [x] Replace callback-stage `HashMap<String, Duration>` mutation with fixed
  thread-local slots. Known stages no longer allocate or materialize `String`
  values per callback; the presentation map is created only on snapshot, and
  unknown labels are retained in an explicit `Callback Other` bucket.
- [x] Populate BVP_Damp residual-request counters at solver callback and
  convergence-residual boundaries, and preserve them in both typed and legacy
  reports. Scalar/FD sub-evaluations remain a separate telemetry scope.
- [x] Add solve-owned elapsed scopes for nonlinear iterations and damping trials
  (2026-09-20). Damped and Frozen now record real iteration durations; Damped
  records each damping-trial duration, while `Off` skips `Instant::now()` and
  all scope writes. Mesh-revision and solve timing continue to come from the
  major typed timers. Worker-thread callback aggregation remains open.
- [ ] Count solver residual/Jacobian requests separately from scalar evaluations,
  chunks, FD internal calls, symbolic analyses, numeric factors and RHS solves.
- [ ] Split conversion, symbolic preparation, callback preparation, callback
  evaluation, matrix assembly, factorization, RHS solve, refinement and AOT
  lifecycle time. State which times are nested; do not sum nested spans twice.
- [x] Report factorization cache hits and invalidations in the typed counters
  and legacy projection. Detailed invalidation reasons and factor generations
  remain follow-up telemetry work. The solver-level Banded gate now observes
  `factorizations=1`, `factorization_cache_hits=5`, `rhs_solves=6` on one
  Jacobian, rather than only exercising the matrix wrapper in isolation.
- [x] Expose actual selected matrix/evaluator/linear algorithm and parameter
  binding policy, not only requested strings. `BvpResolvedPlan` is now attached
  to the typed solver snapshot; before preparation it reports the normalized
  request and after preparation it carries the actual selected evaluator.
- [ ] Report bytes for matrix values, indices, band storage, factors and scratch
  where measurable. Keep dense-equivalent bytes explicitly labeled; use
  unavailable/estimated markers for opaque allocation sizes, never fabricate
  process RSS or report dense-equivalent MiB as actual sparse usage.
- [ ] Record conversions, matrix copies, sparse-pattern builds and output-lock
  acquisitions through opt-in diagnostic/test instrumentation.
- [x] Audit ownership of `CALLBACK_STAGE_TIMERS`: callback stages now use a
  solve-local fixed-slot session, with the old thread-local accumulator kept
  only as a compatibility bridge. Nested-session isolation is covered by a
  debug test; worker-thread aggregation remains open.
- [ ] Remove the current ambiguity between `Symbolic Operations` and
  `Backend Preparation`: `CustomTimer::get_all()` reports the same duration
  under both labels. Use distinct spans for symbolic differentiation,
  callback compilation, backend handoff and generated artifact work.
- [ ] Replace unit-dependent `elapsed_time()` presentation with raw nanoseconds
  or seconds in the typed report. Formatting into `ms/s/min/h` belongs to the
  table/CLI layer and must not change the numeric meaning of a measurement.
- [ ] Make telemetry snapshots immutable and self-describing: solver id,
  solve id, mesh revision, backend plan, lifecycle phase and whether a field is
  measured, estimated or unavailable. A plain `HashMap<String, String>` cannot
  guarantee this contract.
- [ ] Preserve partial reports on failure and atomic parameter-rebind errors.
- [ ] Measure disabled/counts-only/detailed telemetry overhead on small and
  large cases. Counters and stage timing must not add per-entry contention.
  The explicit `BvpTelemetryMode::Off` path now skips counter increments,
  timer starts and callback-stage collection; release overhead evidence is
  still pending.
- [ ] Update story printers alongside schema changes. No production migration
  slice is complete without its diagnostics and semantics tests.
- [x] Keep typed telemetry free of `HashMap` storage (2026-09-20). Counters,
  scopes, timings, and log events use fixed typed fields/slots; the only
  `HashMap` in `telemetry.rs` is the explicitly named `to_legacy_map()`
  compatibility projection. Presentation adapters outside typed telemetry may
  still expose legacy maps until their compatibility contracts are retired.
- [x] Run the release-only `telemetry_off_vs_detailed_adaptive_story` and record
  repeated Off/Detailed overhead. The debug gate already proves adaptive mesh
  refinement, factor invalidation, iteration/trial scopes, and detailed log
  events; release timing is intentionally pending. The latest debug run used
  five interleaved samples: Off median 95.426 ms versus Detailed median
  100.464 ms (about +5.3% for Detailed). The Off maximum of 242.687 ms is an
  outlier, so this is not release evidence and must not be treated as a
  performance claim.

- [x] Enrich the opt-in solver logging contract. Existing `log` output is
  useful but too sparse to explain numerical decisions or performance
  anomalies. Add typed, correlation-ready events for warnings and important
  branches: backend/evaluator selection and fallback, mesh revision and
  refinement, Jacobian refresh/cache invalidation, factorization/cache hits,
  damping coefficient changes and rejected trials, bound-step limiting,
  non-finite/near-singular detection, convergence/termination reason and
  partial-failure context. Typed bounded events now carry solve ids, scopes,
  stages, contextual damping values, backend selection/fallback and
  termination state; the retained trace has an explicit capacity and
  dropped-event count.
- [x] Keep logging and telemetry non-invasive: the default/off path must not
  format strings, allocate event maps or acquire a lock in the numerical hot
  path. `BvpTelemetryMode::Off` skips counters, stage timers and callback
  timing; `BvpLoggingMode::Off` skips event retention independently.
- [ ] Keep detailed logging non-invasive for enabled modes: the default/off path must not format strings,
  allocate event maps or acquire a lock in the numerical hot path. Detailed
  events should carry `solve_id`, `mesh_revision`, `iteration`,
  `damping_trial` and stage identifiers; the enclosing snapshot carries the
  resolved plan. Formatting belongs to
  the selected sink (console, structured test file or tracing adapter).
- [x] Define severity and repetition policy: `debug` for normal adaptive
  decisions, `info` for major lifecycle transitions, `warn` for recoverable
  fallbacks/rejections and `error` for failed termination. Aggregate or rate
  limit repeated scalar/callback events so a large solve does not flood logs.
  Bounded retention now prevents detailed traces from growing with a large
  adaptive solve.
- [ ] Add debug tests for logging-off equivalence, structured event fields,
  damping/mesh/fallback event ordering, failure-context preservation and
  concurrent solves. The first logging-off/order gate is now present; story
  and concurrent-solve coverage remain. Story tests should be able to show
  both the numerical result and the decision trace without changing measured
  solver timings. The logging-off/order, bounded-buffer, failure-event and
  nested solve-local callback gates are present; concurrent-solve story
  coverage remains.
- [ ] Add a release-oriented logging story after the next numerical changes:
  compare `Off`, bounded `Warnings` and bounded `Detailed` modes, including
  dropped-event counts. This remains separate from correctness tests because
  the current instrumentation contract is debug-first.

### 8.1 Typed plan/API migration (debug-first)

- [x] Add read-only `BvpResolvedPlan` for both Damped and Frozen options. It
  normalizes scheme, evaluator policy, symbolic assembly, matrix backend and
  native linear algorithm without preparing equations or mutating a solver.
- [x] Expose `NRBVP::resolved_plan()` on both solver families. Before
  preparation it reports `Auto` where the backend is not selected; after
  preparation it can attach the actual selected backend. This avoids claiming
  that a fallback route was selected merely because it was requested.
- [ ] Move the remaining duplicated Damped/Frozen plan construction behind a
  shared builder/config object. Keep solver-specific convergence/refresh
  policies separate.
- [ ] Make new builder/apply paths fully fallible and keep old mutable/panic
  methods only as compatibility wrappers.
- [x] Add explicit solve-local telemetry ownership for callback stages and keep
  the old thread-local accumulator only as a compatibility bridge. Preserve
  the old presentation adapter until all consumers migrate.

## 9. Tests And Evidence

- [ ] Make the Frozen Criterion harness quiet or route solver INFO output to a
  captured sink. The current bench completes, but per-solve logging floods the
  console and makes the timing summary difficult to audit.

Reuse the following current gates, extending them only where the changed
contract requires it. Preserve test discovery counts/paths when moving code.

| Existing evidence | Reuse |
| --- | --- |
| `tests/basic_correctness.rs` | Two-point/oscillator/stiff systems, numerical FD/analytical, sparse/banded, adaptive solves. |
| `tests/aot_diagnostics.rs` | `symbolic_assembly_backends_match_two_point_jacobian_on_small_sparse_bundle`, linear-system and banded refinement stories. |
| `tests/aot_race_stress.rs` | Sparse/Banded combustion 1000/3000, callback timing, chunking, build/prebuilt lifecycle. |
| `NR_Damp_solver_frozen.rs` tests | Frozen refresh policies, generated callbacks and prebuilt lifecycle. |
| `BVP_utils_damped.rs` tests | Bound-step sign, feasible-step and boundary behavior. |
| `adaptive_grid_basic.rs` tests | Coarse-grid seq/par parity, endpoints, interpolation and no-refinement identity. |
| `solver_common.rs` tests | Different legacy mesh-count conventions. |

- [x] Add the first multiple-RHS factor-reuse gate for the native banded
  backend. `BandedMatrixType` now shares a lazy `OnceLock` factorization across
  Jacobian clones, and `set_assembly`/`set_solver_config` invalidate it before
  rebuilding. Direct legacy field mutation must call
  `invalidate_factorization`; dense/faer reuse and mesh/backend invalidation
  remain separate follow-up slices.
- [x] Fix and test sparse vector/zero-entry conversion, including both the
  `YEnum` and direct `sprs::CsVec`/faer implementations. Dense conversion now
  reads logical coordinates, so omitted `CsVec` entries remain explicit zeros
  instead of shifting subsequent values.
- [x] Add basic matrix-layout adapter tests for sprs and faer shape and
  row/column coordinates. The vector and basic layout contracts now have
  dedicated regression coverage.
- [x] Define and test sparse matrix triplet policy: duplicate coordinates are
  summed by both adapters, and explicit zero triplets have zero logical value
  without changing the numerical matrix contract.
- [x] Add direct Atom/Banded failed and non-finite callback gates. The direct
  callback now returns `BandedError::NonFiniteCallbackValue` with stage and
  matrix coordinates; singular/factor failures remain covered by the native
  linear-layer gates.
- [x] Propagate the first callback errors through the solver-level `try_*`
  boundary (2026-09-20) for legacy-compatible Dense/faer `Fun`/`Jac`
  callbacks without changing their historical non-fallible trait signatures.
  Panic residual/Jacobian and malformed finite-difference output have focused
  Damped/Frozen/trait tests. Generic callback constructors with richer
  iteration context and panic-hook policy are separate follow-up work.
- [x] Add AtomView Lambdify preflight validation (2026-09-20). Missing or
  mismatched parameter bindings and out-of-range sparse Jacobian coordinates
  now become typed errors before compatibility `Fun`/`Jac` objects are
  installed. Historical callback constructors retain panic wrappers; runtime
  callback ABI conversion remains a separate migration slice.
- [x] Add fixed-CSC structure/zero-crossing and band-slot write parity gates
  (2026-09-20, debug): solver-owned Sparse/Banded callbacks preserve their
  prepared coordinates and slot lengths across ExprLegacy/AtomView rebinding;
  numeric values change and cross-frontend values remain equivalent.
- [ ] Add Damped accepted/rejected trial and Frozen refresh trace comparisons.
- [x] Add repeat-solve, rebind, refinement and interleaved-telemetry tests
  (2026-09-20, debug): the lifecycle modules cover repeated prepared calls,
  numeric/structural invalidation, adaptive refinement and stale-runtime
  rejection. Cross-frontend rejected-trial trace parity remains open.
- [ ] Compare callback values at identical states and tolerances before relying
  on end-to-end convergence. Keep mandatory correctness free of timing limits.
- [x] Add and run the focused numeric assembly allocation/stage benchmark
  `bvp_numeric_assembly` for Dense staging versus direct Sparse/Banded
  triplets at 64/256/512 steps. Record callback time, allocation count and
  bytes before changing node-state ownership.
- [ ] Rerun `bvp_numeric_assembly` after the Jacobian/workspace refactor is
  complete; the 2026-09-19 table is explicitly provisional and must not be
  treated as the final release performance baseline.
- [x] Split the monolithic story ledger into an index plus thematic ledgers:
  correctness/native linear algebra, symbolic preparation, AOT lifecycle, and
  heavy performance. Preserve old entries as undated historical evidence.
- [ ] Re-run every undated story entry after the architecture pass. New and
  rerun entries must record `YYYY-MM-DD`, core count, command, result,
  interpretation and conclusion; a rerun supersedes the old block without
  deleting it.

Planned benchmark/stories (names are provisional, not runnable tests yet):

| Story | Hypothesis and required metrics |
| --- | --- |
| `bvp_linear_factor_reuse_story` | Same Jacobian/multiple RHS: factors vs solves, factor/solve ms, backward error, memory. |
| `bvp_callback_buffer_layout_story` | Same Expr evaluators: legacy vs prepared seq/par, calls, nnz, allocations, bytes, mutex count, callback ms. |
| `bvp_atom_runtime_preparation_story` | Expr runtime vs direct Atom: conversion/differentiation/prepare ms, componentwise parity, warm callback ms. |
| `bvp_numeric_stencil_memory_story` | Analytical and FD assembly: actual matrix backend, local/global calls, peak owned-buffer estimate and allocation bytes. |
| `bvp_damped_frozen_runtime_story` | Existing combustion and endpoint fixtures: total/residual/Jacobian/factor/solve stages and numerical traces. |
| `bvp_telemetry_cost_story` | Telemetry modes, stable solutions/counts and measured overhead. |

- [ ] Use 12 Core release runs as the current machine baseline, at least five
  repetitions, mean/std/min/max or median/min/max, with runtime/toolchain and
  thread settings recorded. Keep old 4 Core results historical.
- [ ] Compare ownership changes and arithmetic-representation changes in
  separate rows. Avoid dense nonlinear benchmarks as a proxy for BVP sparse
  assembly. Warm AOT comparisons must exclude compilation explicitly.
- [ ] Extend existing story sections when they already answer the question.
  New sections need test name, debug/release commands, hypothesis, recorded
  result, interpretation and conclusion; never invent unrun results.

## 10. Legacy Preservation And Proposed Module Boundaries

- [x] Introduce explicit `symbolic::bvp::{legacy,direct,telemetry}` import
  boundaries. The old public `symbolic_functions_BVP` path remains available,
  while new direct BVP code has a named home and must not import a legacy
  callback by accident.
- [x] Attach lock-free typed telemetry to the direct Banded Jacobian runtime
  from its first public runtime constructor: callback count, evaluated work,
  elapsed callback time and rejected calls are available without a telemetry
  `Mutex`. The compatibility closure constructor may intentionally discard the
  handle, but new solver integration must retain it.
- [x] Physically move the implementations previously hosted by
  `symbolic_functions_BVP.rs` and `symbolic_functions_BVP2.rs` into
  `symbolic::bvp::{legacy,direct}`. The old filenames are now compatibility
  re-export facades, so downstream imports remain stable.
- [x] Detach direct Banded methods from the legacy `Jacobian` host and give the
  direct module its own `DirectBandedProblem` prepared-problem owner. The old
  `Jacobian` methods remain thin adapters that create an owned snapshot; direct
  preparation, callback execution and telemetry no longer run as methods on a
  legacy type.
- [x] Add a separate `legacy_lambdify` module for the ExprLegacy Dense/faer
  callback builders and lock-free callback telemetry. The historical public
  `Jacobian::lambdify_*` methods remain compatibility adapters.
- [x] Move ExprLegacy symbolic differentiation, sparse-entry construction and
  bandwidth discovery into `symbolic::bvp::legacy_symbolic`; preparation is
  now independently identifiable from callback compilation/evaluation.
- [x] Keep the public `Jacobian` differentiation methods as compatibility
  wrappers while routing optimized, bandwidth-aware, smart, and full parallel
  ExprLegacy differentiation through `legacy_symbolic`. The historical public
  names remain stable and the correctness reference is retained during cleanup.
- [x] Add an Atom-native Lambdify callback branch for Dense/faer Sparse. With
  `BvpSymbolicAssemblyBackend::AtomView`, residual atoms and packed sparse
  derivative atoms are compiled directly; the Expr cache remains only for
  compatibility and the generated-backend bridge.
- [x] Move the shared Lambdify callback counter schema to `bvp::telemetry` and
  keep separate handles for ExprLegacy and AtomView, so stage stories cannot
  mix their runtime costs.
- [x] Physically separate AtomView callback builders into `atom_lambdify`;
  `legacy_lambdify` now contains only ExprLegacy callback compatibility code.
- [x] Expose typed symbolic-preparation stage telemetry alongside the historical
  timing map: variable-set discovery, row differentiation, dense-cache
  materialization and sparse-cache flattening are now comparable without
  parsing string keys.
- [ ] Extend the same direct-owner boundary to the remaining direct Atom/AOT
  preparation APIs. The current change covers the no-Mutex Banded Lambdify
  runtime; Atom/AOT ownership must be migrated without reintroducing a legacy
  callback dependency.
- [x] Extend Atom-native Lambdify to the native Banded callback. The direct
  Banded owner now accepts packed residuals and sparse derivative atoms, keeps
  the compatibility `Expr` payload separately, and evaluates parameters once
  per callback rather than once per nonzero entry.
- [x] Add a parameterized Banded Atom correctness gate covering the native
  `[parameters..., unknowns...]` evaluator ABI and residual/Jacobian parity.
- [x] Rerun and record the Banded ExprLegacy/AtomView release comparison on the
  12 Core machine. The process-isolated rerun keeps cold preparation separate
  from callback timings and confirms that the lazy Atom->Expr compatibility
  bridge restores `AtomView+tcc` AOT without reintroducing eager Expr materialization.
- [x] Audit BVP `.simplify()` calls in the symbolic preparation path. The
  ExprLegacy discretizer no longer simplifies each residual before applying
  boundary conditions and then simplifies the same row again; it performs the
  required final simplification once after BC substitution. The remaining
  Expr simplification calls are intentional legacy/reference paths or the
  singular `t == 0` compatibility fallback and require separate parity work
  before removal.
- [ ] Introduce one typed prepared runtime owner for Dense, faer Sparse and
  Banded. The current Dense/faer/Banded implementations still adapt through
  public legacy trait-object output types at the solver boundary.
- [ ] Add parity and stage stories comparing, at identical states, ExprLegacy
  and AtomView differentiation plus ExprLegacy and Atom-native Lambdify
  callbacks. Record preparation, callback compilation, residual and Jacobian
  runtime telemetry separately.
- [ ] Measure Atom evaluator input/result allocations before introducing
  reusable evaluator workspaces. Do not claim Atom-native callbacks are faster
  until a release benchmark demonstrates it.
- [ ] Retain an explicit legacy reference selection for comparisons and keep
  the reference implementation compiling and exercised permanently.
- [ ] Keep legacy implementations compiling and exercised permanently. Preserve
  known-correct numerical behavior, while recording known legacy defects and
  limiting oracle claims to validated domains.
- [ ] Keep old and new callback/linear implementations independently callable
  in tests. A wrapper that redirects both comparison routes to the new code
  cannot serve as a regression oracle.
- [ ] Prefer focused internal modules as work lands: resolved plan, mesh/layout,
  callback preparation, numeric assembly, linear runtime, workspace, telemetry,
  and legacy adapters. Do not perform a cosmetic file split ahead of behavior
  and ownership changes.
- [ ] QoL module split: after the current no-AOT ownership gates are green,
  move the private Damped/Frozen unit-test blocks into child files under
  `tests/`, preserving their module privacy and stable test names. Split
  production orchestration only along tested ownership boundaries; do not
  rewrite solver math merely to reduce line count.
- [ ] Keep generic symbolic runtime pieces under `symbolic`, and existing
  factorization implementations under `somelinalg`. BVP owns their composition.

## 11. Execution Order And Exit Criteria

1. Establish focused baselines and typed telemetry for callback, allocation,
   matrix-copy and factor/solve work; map existing tests to changed contracts.
2. Introduce factor ownership/invalidation with the current evaluators and
   existing linear algorithms, preserving Damped/Frozen numerical decisions.
3. Add reusable callback/work buffers and built-in typed matrix adapters;
   remove dense/sparse output mutexes while keeping scalar Expr evaluation.
   Every new callback must expose typed telemetry in the same change; do not
   retrofit diagnostics after performance work has landed.
4. Integrate the shared direct Atom evaluator through the BVP adapter, including
   separate preparation and callback benchmarks and parameter semantics.
5. Remove dense staging from numerical structured assembly; introduce structured
   FD only after the analytical stencil path is verified.
6. Complete fallible runtime propagation, resolved-plan compatibility and
   solve-local reports as each path migrates. Isolate the replaced legacy code;
   no new/direct and legacy callback implementation may remain in one runtime
   selection branch.
7. Re-run existing BVP release stories, update guides and evidence, and consider
   defaults separately. Then reuse the shared symbolic runtime in LSODE2.

- [ ] Exit gate: no unexplained convergence/iteration/refinement regressions;
  mandatory callback/layout/cache/error tests pass.
- [ ] Exit gate: factorization reuse is demonstrated by counts, not inferred
  from wall-clock time; no stale factors survive invalidation.
- [ ] Exit gate: sparse/banded callbacks avoid compulsory dense materialization;
  prepared into-buffer paths avoid repeated structural allocation where native
  library APIs permit it, with any remaining allocation measured and labeled.
- [ ] Exit gate: telemetry, user configuration, legacy oracle and AOT lifecycle
  stay usable throughout the migration.
- [ ] Exit gate: no claimed performance gain until recorded release evidence.

## 12. Pure Lambdify AtomView Discretization (2026-09-19)

Scope is deliberately limited to the pure Lambdify path. AOT preparation,
artifact lifecycle and Lambdify-vs-AOT stories are out of scope for this slice.

- [x] Add an internal `AtomDiscretizationInput` that owns converted RHS atoms,
  interned `Symbol` names, indexed node symbols and precomputed per-node rename
  maps.
- [x] Add an internal `AtomBvpProblem` that binds the symbol-only input to a
  mesh and step sizes.
- [x] Keep the public `Vec<Expr>` entry point as a compatibility wrapper that
  performs `expr_to_atom` once before parallel row assembly.
- [x] Move the former `t == 0` assembly fallback onto the Atom-native path.
  Raw Atom `Add/Mul/Pow` construction preserves a singular `0^-1` structure
  instead of evaluating it during coefficient normalization.
- [x] Add a correctness gate for the singular zero endpoint, alongside the
  existing ExprLegacy parity tests for Lane-Emden, two-point BVP and fractional
  coefficients.
- [x] Keep internal symbol identity in `Symbol`; materialize stripped names
  only in the public `DiscretizedBvpAtomSystem` boundary and avoid a
  `String -> Symbol` round trip in the consistency check.
- [ ] Define and document the solver-facing policy for evaluating a singular
  endpoint. Assembly now preserves the exact structure, but regularization or
  a limiting boundary value remains problem-specific.
- [ ] Add a prepared Atom-only public/internal constructor for callers that
  already own packed atoms, so the compatibility `Vec<Expr>` wrapper is not
  required by the production Lambdify route.
- [ ] Re-run the BVP correctness and release stage stories after the complete
  Atom-native migration; current timings are not a final performance baseline.

## 13. AtomView Warm-Callback Regression Audit (2026-09-19)

- [x] Localize the warm Jacobian regression to the Atom `PreparedEvaluator`:
  every scalar callback evaluation allocated a fresh node-result vector and
  temporary evaluation state.
- [x] Add a thread-local reusable evaluator workspace. This keeps the public
  allocating evaluation API unchanged while making prepared Lambdify callbacks
  reuse scratch buffers independently on each Rayon worker, without a Mutex.
- [x] Re-run the existing release Banded story on the same workload. AtomView
  Jacobian improved from `0.966 ms` to `0.470 ms`; warm residual stayed at
  `0.161 ms`, and total solve time is `4.959 ms` versus `4.741 ms` for the
  ExprLegacy oracle. The previous multi-fold Jacobian regression is therefore
  fixed; the remaining small gap is evaluator arithmetic/adapter cost, not
  linear-system or cold-preparation overhead.
- [x] Repeat the same release story five times to separate noise from a
  systematic difference. Warm totals were ExprLegacy
  `4.337 +/- 0.114 ms` and AtomView `4.684 +/- 0.137 ms`: AtomView is about
  `0.347 ms` (`8.0%`) slower on average, but the single-run difference varies
  from `0.092` to `0.592 ms`. This is a small residual cost with substantial
  run-to-run noise, not a new multi-fold regression.
- [ ] Keep the release story as the gate for further evaluator optimization.
  Add opt-in stage telemetry only if the remaining Jacobian gap needs another
  decomposition; do not put per-node timers into the normal callback hot path.

## 14. Prepared Atom-Only Discretization Entry Point (next)

- [x] Add an Atom/Symbol entry point that accepts already converted equations,
  value symbols, argument symbol and typed boundary-condition symbols without
  forcing the caller through `Vec<Expr>` or string-to-symbol conversion.
- [x] Keep the existing `Vec<Expr>` function as a compatibility wrapper and
  cover parity between both entry points with a correctness test.
- [ ] Route the production BVP preparation owner through this entry point once
  the surrounding solver handoff no longer needs the compatibility wrapper.

## 15. Synchronized Correctness And Telemetry (2026-09-19)

The Atom-native discretization result now carries a typed stage snapshot in
parallel with the production code change. This keeps correctness tests and
diagnostic semantics synchronized instead of adding telemetry as an afterthought.

- [x] Add `BvpAtomDiscretizationTelemetrySnapshot` with precise `Duration`
  fields for boundary handling, equation discretization, boundary application,
  flattening, consistency checks, bounds/tolerances, and total time.
- [x] Keep `timer_hash` as a compatibility/presentation projection; it is not
  the primary representation for new tests or hot-path decisions.
- [x] Add correctness coverage for typed stage invariants and for both the
  `Vec<Expr>` compatibility wrapper and the Atom/Symbol entry point.
- [x] Extend the same typed snapshot through the production solver handoff so
  solver-level stories can correlate preparation, callback, and linear-solve
  stages without parsing maps or rendered tables.
- [x] Add matching Damped/Frozen solver-level regression coverage so typed Atom
  preparation telemetry cannot silently disappear from one solver family.
- [ ] When the remaining Atom-native migration is complete, rerun the dated
  release performance gate and compare stage telemetry against the historical
  ExprLegacy oracle; correctness tests remain debug-first, while performance
  stories remain release-only.

## 16. Production Lambdify Telemetry Contract (2026-09-19)

This slice is deliberately limited to the pure Lambdify route. AOT lifecycle,
artifact preparation and Lambdify-vs-AOT performance stories are not part of
this contract.

- [x] Provide one shared `BvpLambdifyTelemetryMode` for ExprLegacy and
  AtomView callbacks: `Off`, `Counters` and `Detailed`.
- [x] Make `Off` the solver default. Disabled callbacks hold no `Arc`, do not
  call `Instant::now()`, and do not touch atomics on the hot path.
- [x] Keep `Counters` cheap: it records only relaxed residual/Jacobian request
  counters. `Detailed` adds elapsed callback time through the same lock-free
  handle, without a per-entry `Mutex` or allocation.
- [x] Keep ExprLegacy and AtomView streams separate, while exposing the same
  typed snapshot schema. The snapshot includes its collection mode so zero
  duration cannot be confused with an unmeasured field.
- [x] Carry live callback handles through Damped/Frozen solver handoff and
  take immutable snapshots only at the solver statistics boundary. A snapshot
  taken during preparation must not be detached from later callback updates.
- [x] Expose the mode through the BVP solver options and provide a thin setter
  for callers that configure an already-created Damped/Frozen solver.
- [x] Cover mode semantics, callback recording and handoff survival with
  correctness tests. Keep the existing compatibility constructors and
  allocating legacy report projections unchanged.
- [ ] Add a release-only overhead story for all three modes on representative
  Dense/faer/Banded Lambdify workloads. Report callback wall time separately
  from solver wall time; do not infer overhead from one noisy run.
- [ ] Extend the typed snapshot with solve/iteration scope identifiers and
  explicit measured/estimated/unavailable stage status when the surrounding
  solver telemetry contract is ready. Do not put strings or maps in callback
  recording merely to satisfy that future schema.

Production rule: new pure-Lambdify callback code must use this shared mode and
live-handle contract. It must not create ad-hoc timers, maps or locks, and it
must add its correctness/telemetry gate in the same change. AOT remains a
separate later migration.

## 17. External Story-Test Reports (2026-09-19)

- [x] Add Utils::test_reporting::write_test_report for dated, canonical
  per-test Markdown reports with safe cross-platform names.
- [x] Replace an existing report for the same suite/test instead of appending
  stale runs; write through a temporary sibling so readers do not observe a
  partial file.
- [x] Keep report I/O outside measured solver scopes. A story must finish all
  timing samples before constructing and writing its report.
- [x] Exclude the repository-local test_reports directory from Cargo package
  inclusion; allow RST_TEST_REPORT_DIR for CI or an external report folder.
- [x] Integrate the first report into
  frozen_dense_faer_banded_runtime_story.
- [x] Integrate the pure-Lambdify Banded ExprLegacy-versus-AtomView story
  without adding report I/O to its cold, callback or solve timers.
- [x] Run that story in release on the combustion-1000 fixture and write the
  current snapshot. The latest report records exact parity
  (residual 4.44e-16, Jacobian 8.67e-19, solution 8.88e-16), ExprLegacy
  cold setup 366.127 ms versus AtomView 193.782 ms, and warm solve
  8.159 ms versus 7.465 ms.
- [x] Preserve historical STORY TESTS ledger rows unchanged. New reports are
  dated observations to compare against the archive, not replacements for it.
- [x] Integrate reports into the verbose pure-Lambdify parity corpus and the
  telemetry price story (2026-09-20). Report creation runs only after all
  assertions and measured samples have completed; quiet correctness tests do
  not create empty report files.
- [ ] Migrate the remaining verbose BVP story tables (backend comparison,
  AOT diagnostics, lifecycle and matrix stories) to the same report contract.
- [ ] Add corresponding suite-specific report paths for BVP_sci, LSODE2 and
  Nonlinear systems without changing their solver timing scopes.

## 18. Delayed Factor-Owner Release Audit (2026-09-19)

- [x] Re-run the complete BVP_Damp debug gate after the Dense/faer/Frozen
  factor-owner and AtomView dispatch changes: 184 passed, 0 failed, 44
  ignored.
- [x] Re-run the complete BVP_Damp release gate: 184 passed, 0 failed, 44
  ignored, 31.39 s after compilation.
- [x] Re-run the focused release factor-runtime tests: 3 passed, 0 failed.
- [x] Re-run the Frozen Dense/faer/Banded release story: 1 passed, 0 failed;
  all routes kept finite solutions, one factorization and one cache hit.
- [x] Re-run the parallel Rust AOT exact-example acceptance: 1 passed, 0
  failed after regularizing the Lane-Emden test fixture away from its
  removable x=0 singularity.
- [ ] Establish a matched historical release baseline before claiming a
  performance improvement; current results are correctness and lifecycle
  gates, not a final ranking.

## 19. Prepared Numeric Parameter Binding Across Matrix Backends (2026-09-20)

The intended prepared contract is the same as the Nonlinear systems route:
symbolic equations are discretized and differentiated once, residual/Jacobian
evaluators are compiled once, and subsequent solves only replace numeric
parameter values. Parameters are evaluator inputs, not Newton unknowns and are
not differentiated. Rebinding must update callback values while invalidating
numeric Jacobian/factor state when the next solve requires it.

- [x] Add a shared typed numeric binding handle whose replacement is separate
  from symbolic preparation. Callback workers take one immutable value snapshot
  per request; no lock is held while assembling a Jacobian.
- [x] Cover Dense ExprLegacy and AtomView callback rebinding without rebuilding
  the symbolic evaluator.
- [x] Cover faer Sparse ExprLegacy and AtomView callback rebinding, including
  residual/Jacobian value changes and sparse layout preservation.
- [x] Cover the native Banded direct runtime, including packed callback ABI,
  residual/Jacobian parity and numeric replacement after compilation.
- [x] Represent prepared identity through the common typed `BvpPreparedPlan`
  metadata and preserve a typed stale-plan boundary in Damped/Frozen.
- [x] Add the first Frozen `try_solver_prepared` entrypoint; stale prepared
  state is reported as a typed invalidation error instead of silently reusing
  callbacks or factors.
- [x] Add one solver-surface acceptance story that prepares a parameterized
  Damped BVP and repeats solves through nalgebra Dense, faer Sparse and native
  Banded without `try_eq_generate` between bindings (debug gate added
  2026-09-20).
- [ ] Measure cold preparation versus warm rebind in release on the same
  problem and record callback, factorization and solve stages separately.
- [ ] Add an explicit factor-invalidation assertion to the solver-level story:
  rebinding may preserve symbolic/Lambdify artifacts, but must not reuse a
  numeric factorization built for the previous parameter values.
- [ ] Decide and document the public prepared API for initial parameter values,
  parameter-name ordering, missing values and atomic rebind failure. The typed
  path must not expose the internal synchronization primitive.

## 20. P0: AtomView Parallel Policy, Prepared Ownership And Lifecycle (2026-09-20)

This is the next correctness-first block for the pure Lambdify route. AOT is
explicitly out of scope here. The existing AtomView callbacks already use
parallel evaluation in several places, but the policy is implicit and the
prepared runtime is still distributed between solver fields, callback bundles,
revision metadata and solver-local factor owners.

### AtomView execution policy

- [x] Introduce one explicit internal execution policy for AtomView Lambdify
  (2026-09-20):
  `Sequential` and `Parallel { min_work: usize }`. The policy must apply to
  pure callback residual/Jacobian evaluation for Dense, faer Sparse and
  native Banded routes; symbolic preparation remains independently parallel.
  Existing callback entry points preserve the historical default
  `Parallel { min_work: 0 }`, while policy-aware builders expose an explicit
  sequential route. AOT is not changed.
- [x] Make the sequential path genuinely sequential and the parallel path
  genuinely observable through typed dispatch diagnostics (2026-09-20,
  debug). Do not infer policy from Rayon defaults or from the presence of
  `par_iter()` alone; the solver-level gate now observes the selected branch.
- [ ] Keep the no-Mutex guarantee: parallel Jacobian workers must write to
  disjoint row/column/diagonal/value ranges or to explicitly owned temporary
  ranges, never to a shared locked matrix.
- [x] Add debug callback-level correctness gates (2026-09-20) proving
  Sequential and Parallel produce componentwise-identical Dense/Sparse/Banded
  residuals and Jacobians. Fixed CSC coordinate stability and full solver
  trace parity remain in the later lifecycle gate.
- [x] Add a solver-level pure-Lambdify policy gate (2026-09-20, debug) for
  ExprLegacy and AtomView on faer Sparse and native Banded routes. It checks
  componentwise solution parity and typed sequential/parallel dispatch
  counters; it makes no performance claim and does not exercise AOT.
- [ ] Add a release-only break-even story for Dense, faer Sparse and Banded
  after the correctness gate. Report work size, selected policy, worker/chunk
  count, residual/Jacobian time and allocation/copy stages separately.
- [ ] Cover the current gaps explicitly: runtime Dense Jacobian evaluation,
  Sparse triplet-to-CSC assembly, Banded EntryChunks scatter and per-request
  residual/Jacobian output allocation. Do not call the AtomView path fully
  parallel or fully allocation-free until these are measured.

### P0 prepared-plan ownership

- [ ] Finish the common `PreparedPlan` as the owner of prepared runtime state,
  not only revision/fingerprint metadata. It must own or reference the
  prepared mesh/layout, callback bundle, Jacobian generation and factor
  generation through one typed boundary.
- [ ] Keep Damped and Frozen numerical policies separate while sharing the
  prepared ownership and invalidation machinery. Compatibility fields may be
  projected into the plan, but the solve loop must not assemble its runtime
  from unrelated mutable fields after preparation.
- [ ] Define the plan lifecycle explicitly: `Unprepared`, `Prepared`,
  `NumericalJacobianCurrent`, `FactorCurrent` and `Invalidated`, with typed
  transitions and no silent fallback to stale callbacks or factors.

### P0 invalidation matrix

- [ ] Close and test invalidation for parameter values/names, mesh, boundary
  conditions, state values, solver policy, matrix backend, evaluator policy,
  Jacobian structure and bandwidth/pattern changes.
- [ ] Prove that numeric parameter rebinding preserves symbolic/Lambdify
  artifacts but never reuses a factor built from the previous numeric
  Jacobian.
- [ ] Prove that mesh, BC, layout, backend or Jacobian-pattern changes discard
  both structural analysis and numeric factors before the next solve.
- [x] Add Damped and Frozen Sparse/faer+Banded interleaved lifecycle gates
  (2026-09-20, debug): prepare, solve, numeric rebind, mesh/BC/policy changes
  where the solver exposes them, direct public compatibility-field mutation,
  typed stale rejection and regeneration. Dense-control and full
  pattern/evaluator-policy coverage remain open; Frozen has no public mesh
  setter, so its mesh branch is intentionally not claimed here.

### P0 parity and typed failure boundary

- [ ] Extend ExprLegacy/AtomView parity beyond final solutions: callback
  values at identical states, accepted and rejected Newton traces, damping
  decisions, refinement traces, fixed-CSC coordinates/zero crossings and
  Banded row/column slot writes.
- [x] Add solver-level residual shape and non-finite callback gates
  (2026-09-20, debug), including the public `try_calc_residual` boundary and
  report-file evidence.
- [x] Add the public typed Jacobian callback boundary
  (2026-09-20, debug): malformed prepared Jacobian output is surfaced as
  `CallbackShapeMismatch` through `try_recalculate_jacobian` rather than
  escaping as a compatibility panic. Singular-factor, invalid-layout and
  partial factor/cache telemetry gates remain open.
- [ ] Move callback and linear failures to a typed `try_*` boundary. Existing
  infallible `Fun`/`Jac` compatibility wrappers may remain, but new prepared
  callbacks must not expose `panic!`, `unwrap` or `expect` for malformed user
  input or runtime shape/type failures.
- [ ] Keep compatibility panic methods explicitly marked as wrappers and
  ensure they cannot be selected accidentally by the new prepared runtime.

Exit condition for this P0 block: the prepared owner, invalidation matrix,
AtomView Sequential/Parallel policy, parity traces and typed failure tests are
green in debug. Only then should the release break-even and allocation stories
be used to choose the default execution policy or claim a performance gain.

## 24. Cross-product Lambdify correctness corpus (2026-09-20)

The previous parity tests were intentionally centered on the production Banded
route. That was not enough to detect a frontend/backend-specific regression.
The new `test_lambdify_cross_product` module is the correctness matrix before
any further optimization:

- [x] Compare `ExprLegacy` and `AtomView` callback residuals, dense Jacobian
  values and final solutions on the same oscillator fixture across Dense,
  faer Sparse and native Banded (debug, 2026-09-20).
- [x] Mark Dense explicitly as a control route; performance conclusions must
  use faer Sparse and native Banded only.
- [x] Compare native Banded `Sequential` and `Parallel { min_work: 0 }`
  evaluation with both `Diagonal` and `EntryChunks` decomposition on a
  nonlinear four-variable corpus (debug, 2026-09-20).
- [x] Preserve Banded out-of-storage semantics: reads outside `(kl, ku)` are
  mathematical zeroes, while in-band slot values must match exactly.
- [x] Write canonical reports outside measured code to
  `test_reports/bvp_damp/` for both verbose tests.
- [x] Add the same explicit execution-policy selection to the solver-level
  Sparse and Banded configuration, then repeat this matrix with policy and
  dispatch diagnostics visible in the report (2026-09-20, debug).
- [ ] Extend the fixture set with combustion-1000 and one independent stiff or
  nonlinear BVP before accepting any release performance ranking.
- [x] Add the first Sparse/faer fixed-CSC coordinate/order gate (2026-09-20,
  debug): ExprLegacy and AtomView preserve `col_ptr`/`row_idx` through numeric
  rebind and publish matching values. Factor invalidation and the analogous
  Banded slot gate are now covered by the paired debug lifecycle test.
- [x] Make the AtomView faer Sparse Lambdify callback prepare a fixed CSC
  pattern (2026-09-20, debug). Numeric callbacks now evaluate only values and
  publish against a prepared `col_ptr`/`row_idx` pattern; the current owning
  callback ABI still clones that symbolic structure when returning a matrix.
  Duplicate coordinates retain the legacy triplet fallback because faer sums
  them. The zero-crossing structure gate passes; release speedup remains
  pending. Full zero-allocation owner reuse is a separate follow-up.
- [ ] Possible technical debt: introduce an owner-level Sparse callback/runtime
  boundary that can reuse one CSC symbolic owner and numeric value buffer
  without cloning structure on every returned `SparseColMat`. This requires a
  deliberate MatrixType/Jac ABI decision and must not be solved with a Mutex
  in the production hot path.
- [x] Add the solver-level native Banded slot/factor gate (2026-09-20, debug):
  diagonal offsets and slot lengths remain stable across numeric rebind,
  ExprLegacy/AtomView values match, and telemetry proves invalidation followed
  by a new factorization. This does not yet claim common factor ownership.

Debug command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_cross_product -- --nocapture --test-threads=1
```

This corpus is a correctness gate, not a speed claim. The release stage
baseline remains the separate ignored story, and AOT is intentionally outside
both tests.

## 21. Typed Matrix Backend Compatibility Gate (2026-09-20)

The public compatibility surface still accepts historical string selections
such as `Dense`, `Sparse` and `Banded`. They are intentionally retained for
task documents and older callers. New typed callers must not have to encode a
backend as a string.

- [x] Verify the closed `MatrixBackend` enum is available from the BVP_Damp
  public surface and can select Dense, Sparse/faer and Banded without writing
  the legacy `method: String` field.
- [x] Add typed `with_matrix_backend(...)` builder methods for Damped and
  Frozen options. The resolved typed plan is the source of truth; the legacy
  string remains a compatibility projection rather than a second independent
  configuration path.
- [x] Add a debug gate covering typed selection, legacy string round-trip and
  resolved-plan precedence. This proves API compatibility, not runtime
  factor-owner readiness.
- [ ] Add the same typed-selection assertions to task-parser and prepared
  solver acceptance tests, including invalid/unknown string diagnostics.

## 22. Lambdify Release Baseline Before P1 (2026-09-20)

This is the release gate before changing linear-runtime ownership or callback
assembly. It is deliberately Lambdify-only; AOT tests and AOT-vs-Lambdify
stories are not part of this baseline.

- [x] Run the complete pure-Lambdify release set for ExprLegacy and AtomView,
  including symbolic callback tests, direct/native Banded tests, parity corpus,
  parameter rebind, resolved-plan tests and the telemetry price story
  (2026-09-20; all reported release gates passed). The canonical stage report is
  `test_reports/bvp_damp/lambdify_stage_baseline_corpus.md`.
- [x] Add the debug cross-product correctness gate for ExprLegacy/AtomView x
  Dense-control/Sparse-faer/Banded and Banded Sequential/Parallel layouts
  (Section 24, 2026-09-20).
- [x] Implement and debug-smoke the expanded stage baseline story on
  Sparse/Banded combustion, Sparse/Banded oscillator and the small Dense-control
  oscillator (`runs=1`, 2026-09-20). Dense is intentionally not used for large
  workloads.
- [x] Run the expanded stage baseline in release with `runs=3` and record
  dated total, residual, Jacobian, factorization, RHS-solve timings, counters,
  factor/cache counters and numerical parity (2026-09-20). Preserve the
  historical ExprLegacy rows; do not overwrite them with a new baseline.
- [x] Check the current ExprLegacy rows against the retained historical route
  records and require every AtomView claim to name its stage and workload. The
  current data shows no clear ExprLegacy regression, while all AtomView cold
  preparation claims are stage-qualified. A wall-clock change without stage
  telemetry remains invalid evidence.
- [x] Keep the stage-baseline solution drift guard separate from callback
  correctness: production routes and the Dense control are checked against an
  explicit `1e-6` diagnostic limit, while strict callback/Jacobian parity stays
  in the debug cross-product gate. The smoke run observed `3.27e-7` on the
  oscillator production routes and `4.78e-7` on Dense-control.
- [ ] Investigate the measured warm AtomView regression on combustion-3000
  Banded (`16.413 ms` solve wall versus `13.088 ms` ExprLegacy), especially
  Jacobian callback (`1.755 ms` versus `0.796 ms`) and solver Jacobian time
  (`1.878 ms` versus `0.441 ms`), before optimizing unrelated stages.
- [ ] Add fixed-CSC, band-slot and rebind rows to the same release baseline now
  that their debug correctness gates are green. Release numbers remain
  provisional until the full P0 lifecycle corpus is complete.

## 23. P1: Shared Linear Runtime And AtomView Hot Path (after P0)

Do not start this block until the P0 debug exit condition and Section 22
release baseline are complete. The target production backends are nalgebra
Dense, faer Sparse and the native Banded solver. Experimental `sprs` and other
legacy implementations remain compatibility/non-production lanes.

- [ ] Introduce one closed internal `BvpLinearRuntime` dispatcher with native
  Dense/faer/Banded variants; keep the public callback extension boundary open.
- [ ] Split structural analysis, numeric factorization and `solve_into(RHS)`
  into explicit stages. Reuse existing backend factor implementations rather
  than adding a second LU implementation.
- [ ] Move factor ownership from solver-local prototypes into `PreparedPlan`
  and prove factor reuse/invalidation with counts, generations and typed
  telemetry, not wall-clock inference.
- [ ] Add reusable buffers for residual, current/candidate state, Newton step
  and matrix assembly. Measure allocations/copies before and after each slice.
- [ ] Remove mandatory dense staging/conversions from Sparse and Banded paths;
  audit Triplet creation, duplicate-coordinate handling and fixed-CSC writes.
- [ ] Keep AtomView production-native: `Expr` conversion is allowed only at an
  explicit legacy/AOT adapter boundary. Add stage telemetry for any remaining
  conversion, copy or materialization.
- [ ] Benchmark AtomView Sequential/Parallel/Auto policies with multiple
  chunking strategies. The deterministic Auto policy is now implemented; if
  thread-startup calibration is added, isolate and report it rather than hiding
  it inside callback timings.
- [ ] Add release stories for Dense, faer Sparse and Banded with residual and
  Jacobian break-even points, chunk counts, allocations, copies and numerical
  parity. Do not promote Auto or Parallel by default without this evidence.

## 25. Pure Lambdify Acceptance Matrix (2026-09-20)

This section is deliberately limited to the Lambdify route. AOT preparation,
external compilers and AOT-vs-Lambdify comparisons remain outside this matrix.
Historical ExprLegacy story rows are evidence and must be preserved; new rows
are dated control data for the AtomView migration.

### 25.1 Correctness and analytical solutions

- [x] Keep the existing linear two-point and oscillator analytical fixtures in
  `parity_corpus` as debug correctness gates.
- [x] Keep the small `ExprLegacy`/`AtomView` callback cross-product gate across
  Dense-control, faer Sparse and native Banded.
- [x] Add a common exact-solution corpus for Linear, Oscillator, one nonlinear
  problem with a known analytical solution and one variable-coefficient BVP.
  The debug gate now includes the stronger Bratu-like adaptive fixture.
- [x] For the current nonlinear and variable-coefficient production fixtures,
  check analytical max-error bounds, endpoint BC values, finite published
  states, and bounded finite discrete residuals on the reduced Newton state;
  the residual bound is explicitly discretization-level, not a timing claim.
- [x] Add a genuinely nonuniform analytical mesh fixture; the pure-Lambdify
  gate now checks custom mesh preservation, exact node values and parity for
  `ExprLegacy/AtomView x Sparse/Banded`. Adaptive refinement is covered
  separately by the Bratu-like gate.
- [x] Add solver-level non-finite, singular-linear-system, invalid-layout and
  callback-shape failure tests with typed errors and partial telemetry
  (2026-09-20, debug). Remaining work is to extend the same boundary to every
  compatibility callback, not to add another duplicate fixture.

### 25.2 Frontend and matrix-backend cross-product

- [x] Compare `ExprLegacy` and `AtomView` callback residuals/Jacobians and
  final solutions on the small oscillator cross-product.
- [x] Compare production Sparse/faer and native Banded on combustion-1000 and
  combustion-3000 in the stage baseline; keep Dense as a small control only.
- [x] Repeat the current analytical-solution corpus as
  `ExprLegacy/AtomView x Sparse/Banded`; run Dense only for small control sizes.
- [ ] Add solver-level equality checks for callback values, assembled matrix
  values, accepted/rejected Newton decisions, refinement decisions and final
  residuals for every production route.
- [x] Add an adaptive nonlinear solver-level gate for both frontends and both
  production matrix routes: one DoublePoints refinement, finite published
  state, nonzero iteration trace and final-state parity.
- [x] Add fixed-CSC coordinate/order parity for Sparse and fixed band-slot
  parity for Banded at the solver boundary (2026-09-20, debug); both
  ExprLegacy and AtomView are covered after numeric rebind. Full value,
  decision-trace and all-policy cross-product parity remains open.

### 25.3 Prepared parameters and lifecycle

- [x] Verify numeric parameter rebinding changes Dense/Sparse/Banded callback
  values and invalidates the old numeric factor.
- [x] Extend parameter rebinding to both `ExprLegacy` and `AtomView` across
  Dense/Sparse/Banded; repeated prepared solves remain an additional gate.
- [ ] Prove that rebinding does not repeat symbolic differentiation or
  Lambdification, while factorization and RHS stages use the new values.
- [x] Add the first lifecycle matrix for numeric rebind, mesh, boundary
  conditions, backend policy and direct compatibility-field mutation
  (2026-09-20, debug; Sparse/faer+Banded x ExprLegacy/AtomView).
- [ ] Extend the matrix to parameter-name/layout/evaluator-policy/Jacobian
  pattern changes and full Frozen/Dense-control coverage; the current tests
  are focused slices, not a claim that common `PreparedPlan` ownership is
  complete.
- [x] Add an interleaved prepare/rebind/policy-change/regenerate gate proving
  that stale prepared callbacks are rejected before consumption
  (2026-09-20, debug; the same focused production-route matrix).

### 25.4 Sequential, Parallel and chunking

- [x] Keep low-level Banded correctness coverage for Sequential/Parallel and
  `Diagonal`/`EntryChunks` work decomposition.
- [x] Add solver-level Lambdify tests with explicit Sequential and Parallel
  policies for both Sparse and Banded production routes (2026-09-20, debug).
  The gate compares final states and dispatch counters across ExprLegacy and
  AtomView; a companion test proves `Parallel { min_work: usize::MAX }`
  falls back to sequential without numerical drift.
- [x] Add a release-oriented AtomView Sequential/Parallel break-even story
  (2026-09-20, debug smoke passed): Sparse/Banded and small/combustion
  workloads record cold callback cost, warm callback cost, full solve time,
  integer trajectory counters and solution parity. The current report shows
  Parallel losing on small oscillator work and winning on combustion-1000;
  this is performance evidence for Auto tuning, not a release promotion of any
  particular threshold.
- [x] Cover `min_work` below threshold, exactly at threshold and above
  threshold for the direct Banded callback; solver-level worker reporting
  remains open.
- [x] Add the deterministic `Auto { min_work }` policy gate (2026-09-20,
  debug). It requires at least eight scalar evaluator items per Rayon worker,
  plus the caller's explicit lower bound; small work therefore stays whole/
  sequential while sufficiently large work may dispatch in parallel. The gate
  uses scalar evaluator work rather than the number of Banded diagonals, so a
  wide band is not incorrectly classified as small. This is a correctness
  policy, not a release break-even claim.
- [x] Make `Auto` layout-aware for direct Banded callbacks (2026-09-20,
  debug). The policy now receives both scalar work and the number of effective
  independent tasks. A single long diagonal is split into coarse evaluator
  ranges when it would otherwise expose only one Rayon job; the values are
  still scattered by one owner into native band storage, without a `Mutex`.
  Direct telemetry records the effective task count so story tests can verify
  that a selected parallel branch actually exposed parallel work.
- [ ] Compare chunk strategies by callback time, total solve time, chunk count,
  allocations/copies and numerical parity. Do not infer a speedup from one
  callback measurement.
- [ ] Measure worker/thread startup calibration separately and never include
  it silently in residual/Jacobian timings. The current `Auto` gate deliberately
  does not calibrate on the first callback.

### 25.5 Stage performance and allocation evidence

- [x] Keep the expanded stage baseline for cold setup, callback residual/Jacobian,
  solver residual/Jacobian, factorization, RHS and total solve time.
- [x] Persist integer trajectory counters alongside every stage-baseline row
  (2026-09-20): iterations, residual requests, Jacobian requests/recalculations,
  damping trials/rejections, linear/RHS solves, refinements, factorization
  lifecycle, chunks, conversions and copies. ExprLegacy/AtomView parity now
  fails the baseline if the core numerical trajectory differs.
- [x] Run the complete release baseline with `runs=3` for ExprLegacy/AtomView,
  combustion-1000/3000 Sparse/Banded and the small Dense control (2026-09-20);
  preserve historical rows instead of replacing them. The acceptance policy is
  two-dimensional: ExprLegacy must not regress against its archived release
  baseline; AtomView must remain correct against ExprLegacy and must also not
  regress against its own archived AtomView release rows.
- [x] Add a debug repeated prepared-solve/invalidation gate separate from cold
  preparation (2026-09-20): Sparse/faer and native Banded, ExprLegacy and
  AtomView remain solvable across repeated prepared calls, while numeric rebind
  invalidates and rebuilds the factor. The current API reuses prepared
  callbacks but may refactor at a separate solver-level solve; factor reuse
  across those solves remains an explicit P1 optimization task.
- [ ] Report median/min/max or mean/std consistently for every stage and keep
  callback sampling separate from end-to-end solve samples.
- [ ] Add allocation/copy counters for symbolic preparation, callback assembly,
  triplets, CSC conversion, band scatter, factorization and RHS solve.
- [ ] Require every AtomView performance claim to name the workload, backend,
  execution policy and stage; wall-clock alone is not an optimization result.

### 25.6 Telemetry and canonical reports

- [x] Add a Lambdify-specific detailed telemetry story using symbolic
  ExprLegacy/AtomView callbacks, not only `new_numeric_with_jacobian_options`.
  The debug gate publishes typed generation stages for Sparse/Banded and
  frontend-specific callback streams for Sparse; its canonical report also
  records the current Banded solver-stage-only boundary.
- [ ] Include frontend, matrix backend, evaluator policy, chunking, dimensions,
  structural NNZ, bandwidth, calls, chunks, conversions, copies,
  factorizations, cache hits, invalidations, residual/Jacobian/linear timings
  and numerical error in the canonical report.
- [x] Add `Off`, `Counters` and `Detailed` comparisons proving that disabled
  direct Banded telemetry performs no timing/scope/map work and does not alter
  values (2026-09-20, debug). Solver-owned callbacks now inherit the common
  Lambdify mode; low-level direct constructors remain detailed by default for
  compatibility diagnostics.
- [ ] Keep report writing outside measured regions and use one canonical dated
  report per story test under `test_reports/bvp_damp/`; all current verbose
  pure-Lambdify gates already follow this contract.
- [x] Project the direct no-Mutex Banded callback snapshot into solver-level
  typed statistics (2026-09-20). Banded reports now expose argument,
  evaluator, storage-write, dispatch and elapsed-time stages alongside the
  generic solver Jacobian stage. The handoff retains a cloneable lock-free
  telemetry handle and materializes the immutable snapshot only when solver
  statistics are requested, so it does not add synchronization to the
  callback hot path.
- [ ] Keep historical ExprLegacy records immutable; append new dated baseline
  records after each major runtime change and rerun the complete matrix before
  declaring production readiness.

### 25.7 Debug stage diagnostics before the next optimization pass

- [x] Add fixed-slot, atomic Banded Jacobian telemetry for argument assembly,
  `BandedAssembly` allocation, scalar evaluator work, native storage writes,
  evaluator-call count, storage-write count and selected dispatch/layout.
- [x] Keep diagonal timing semantics explicit: evaluator and direct slot write
  are fused and timed together, while EntryChunks reports evaluator and scatter
  separately. No hot-path `HashMap` or string stage labels are introduced.
- [x] Add direct correctness assertions for diagonal and entry telemetry and
  keep them alongside the no-Mutex callback parity tests.
- [x] Add solver-level nonlinear exact-solution acceptance for both production
  matrix routes and both symbolic frontends; write dated reports outside the
  solver measurement region.
- [ ] After the next Banded optimization, rerun the release stage baseline and
  compare `argument_prepare`, `evaluator`, `storage_write`, `assembly_alloc`,
  callback total and solver Jacobian stages against the 2026-09-20 archive.
- [x] Split the pre-optimization Banded diagnostic into an explicit callback
  sample and a post-solve delta (2026-09-20). The stage baseline now records
  evaluator, storage-write and `BandedAssembly` allocation time separately for
  the sampled callbacks and for Newton callbacks. This is a diagnostic-only
  change; the release baseline must be rerun before interpreting the new rows.
- [x] Optimize native `BandedAssembly` conversion loops (2026-09-20, debug
  validation): `to_banded`, `to_block_tridiagonal` and diagonal fill helpers no
  longer allocate an offsets vector or repeat public `get`/`set` validation for
  every scalar. The storage layout is unchanged and conversion parity remains
  covered by the banded unit tests. This is an accumulated optimization pass;
  no release speedup is claimed until the stage baseline is rerun.
- [x] Remove two more native Banded hot-path copies (2026-09-20, debug
  validation): parallel diagonal fill derives offsets from `enumerate()`
  instead of allocating an `infos` vector, and built-in `DVector` RHS values
  are copied directly into the mutable solver buffer instead of first creating
  a temporary `DVector`. Other `VectorType` implementations retain the typed
  compatibility fallback. The factorization-cache and banded conversion tests
  remain green; release impact is intentionally deferred.
- [x] Remove unconditional direct Banded callback telemetry overhead
  (2026-09-20, debug validation): `Off` now uses an allocation-free disabled
  handle and skips `Instant`/atomic work, while `Counters` skips timestamps and
  `Detailed` preserves the typed stage breakdown. The direct callback and
  telemetry correctness suites passed; release price must be measured together
  with the accumulated Lambdify baseline, not inferred from debug timings.

Exit condition for this section: analytical and callback correctness are green
for both production matrix routes and both symbolic frontends; lifecycle and
typed-error gates are green; Sequential/Parallel/chunking behavior is observed
in solver-level tests; and release reports contain comparable stage timings and
counters. Only then should a default execution policy or AtomView performance
advantage be declared. Dense remains a small correctness control, never a large
problem performance route.

## 26. AtomView Warm Evaluator Investigation (2026-09-20, analysis only)

Scope: pure Lambdify, primarily native Banded, with Sparse/faer as a second
production gate. Implementation is deferred until the diagnosis below is
reviewed. Preserve the ExprLegacy oracle and existing numerical algorithms.

### Evidence and limits

- [x] Inspect the report recorded at `2026-09-20T20:33:28.787Z` (local 23:33),
  `test_reports/bvp_damp/lambdify_stage_baseline_corpus.md`.
  Both Banded frontends report nine parallel sampled Jacobian calls, one
  parallel Newton Jacobian call and zero sequential calls. The configured
  default is `Parallel { min_work: 0 }` for residual and Jacobian. Unequal
  Sequential/Parallel selection does not explain this comparison.
- [x] Locate the actual implementation: both Banded frontends use the direct
  banded assembly machinery in `symbolic/bvp/direct.rs`; ExprLegacy compiles
  recursive Expr closures and AtomView uses `View::PreparedEvaluator` tapes.
  This Banded comparison must not be described as Mutex versus no-Mutex.
- [x] Record the current Banded/3000 evidence: cold setup 782.903/453.723 ms,
  solve wall 13.006/15.652 ms, residual stage 5.317/6.385 ms, Jacobian stage
  0.411/1.796 ms and linear stage 4.307/4.296 ms (ExprLegacy/AtomView).
  Both have 5 iterations, 12 residual calls, 1 Jacobian request, 1 factor,
  9 factor cache hits and 10 linear solves; solution difference is 6.661e-16.
  Cold advantage includes discretization, differentiation and binding; it is
  not a measurement of closure compilation alone.
- [x] Qualify the localization: sampled Banded/3000 evaluator time is
  5.252/10.063 ms across nine calls. In Diagonal mode that timer includes
  evaluator execution, validation, band-slot writes and Rayon scheduling/join.
  A separate storage time of zero means fused work, not free storage writes.
  EntryChunks currently also includes scatter inside evaluator elapsed time;
  sequential EntryChunks storage time includes evaluation. These overlapping
  scopes cannot be added or read as isolated instruction execution costs.
- [x] Check sampling: `runs=3` repeats callback batches only; setup and the
  full prepared solve each execute once per row. Different iteration states,
  warmup and scheduling can explain part of sample/solver timing differences.
  Counts matching do not prove identical accepted/rejected state traces.

### Priority 1: establish the cost model

- [ ] Preserve the 23:33 evidence and earlier dated snapshots before reruns;
  label differences in telemetry mode and measurement scope. Compare AtomView
  with both ExprLegacy and its own previous version, not just one baseline.
- [ ] Extract representative residual and Jacobian expressions from the real
  combustion-1000/3000 and oscillator fixtures. Record preparation-time typed
  summaries: constant/variable-only entries, instruction count distribution,
  Pow exponents (-1, small integers, fractional, variable), builtin counts,
  Add/Mul arities, scratch sizes and work per nonempty diagonal.
- [ ] Benchmark the same mathematical expressions and identical state/parameter
  slices at three levels: scalar evaluator, full callback assembly, prepared
  solve. Include several recorded Newton states as well as the 0.99 sample.
  Warm the Rayon pool and scratch explicitly; report cold calls separately.
- [ ] Repeat complete solves with reset identical initial state, alternating
  frontend order and at least five samples; report median/min/max per stage.
  Label callback repetitions separately from solve repetitions. Compare both
  Sequential and Parallel, then chunk strategies in a separate dimension.
- [ ] Refine timing semantics: explicitly mark fused/overlapping stages or
  introduce non-overlapping boundaries. Add the Banded residual breakdown,
  actual chunk/task count, worker pool size and scratch growth/allocation
  counts. A parallel-dispatch counter does not establish worker utilization.
  Collect static opcode histograms during preparation, not atomically per
  opcode. Use optional coarse worker-local counters and aggregate after work;
  keep Off free of timing/atomic/report work. Write reports outside timings.

### Priority 2: isolated evaluator changes, gated by measurements

- [ ] Investigate Pow specialization first. `evaluate_plain_numeric` calls
  `powf` for every Pow; Atom division is represented via negative powers,
  while Expr has direct division. Measure real opcode frequency and cost
  before choosing reciprocal/small-integer instructions. Validate rounding,
  overflow/underflow, negative bases, signed zero, NaN/Inf and domain errors;
  do not assume `powf`, `powi`, multiplication and division are bit-equivalent.
- [ ] Resolve builtin symbols to typed numeric opcodes at preparation time;
  compare with the current per-evaluation Symbol dispatch. Keep the custom
  function path and its error semantics available.
- [ ] Measure direct Const/Var evaluators for trivial Jacobian entries and
  binary/immediate instructions for short tapes. Preserve operation order;
  avoid broad simplify, reassociation, FMA or cross-expression CSE in this pass.
- [ ] Measure one explicit reusable numeric workspace per job/chunk. Today
  each scalar closure checks arity, enters TLS/RefCell, checks arity again,
  resizes `results`, then evaluates a tape. Repeated short/long tapes can
  reinitialize a grown tail even with sufficient capacity. Retain high-water
  scratch length and overwrite every consumed slot before reading it.
  Validate input ABI once at a typed batch boundary; keep public scalar API
  checks and safe indexing. Cover concurrent calls and reentrant custom
  callbacks; do not replace TLS with shared mutable scratch or a Mutex.
- [ ] If scalar costs justify batching, store typed prepared Atom evaluators
  in the production batch and amortize workspace acquisition and dynamic
  calls across a chunk. Keep scalar compatibility adapters separate. Preserve
  shared variable metadata, prepared numeric rebind and error coordinates.
  Arc capture alone is not evidence of per-call refcount overhead.

### Priority 3: work distribution after scalar costs are known

- [ ] Measure Diagonal load imbalance: its Rayon iterator splits diagonals,
  but each diagonal evaluates its entries serially. NNZ alone does not capture
  different instruction costs; more mesh points do not add diagonal tasks.
  Compare actual per-task workload for both frontends on the same layout.
- [ ] Prototype disjoint subranges within long diagonals, weighted by measured
  or preparation-estimated work, with direct writes to non-overlapping slots.
  Compare against EntryChunks, which currently allocates value triples and
  performs serial scatter. Measure scheduling and scratch/allocation costs.
- [ ] Tune Auto using measured work and available parallel tasks as well as
  pool size. The current eight-items-per-worker gate is a heuristic, not an
  established break-even threshold. Keep Sequential/Parallel explicit and
  include small workloads; do not calibrate secretly on the first callback.
- [x] Add the first layout-aware Auto correctness slice (2026-09-20, debug):
  a 64-entry single-diagonal Banded callback preserves exact values and
  reports eight effective tasks in a two-worker pool. This proves dispatch
  shape, not a performance win; release break-even measurements remain open.

### Acceptance and order

- [ ] First implement the diagnostic corpus and timer-scope corrections;
  then small evaluator specializations; then workspace/batch ownership; then
  partitioning. Keep each change independently attributable in saved results.
- [ ] Run debug correctness for scalar math/domain cases, batch/scalar parity,
  fixed band slots and CSC, parameter rebind, concurrency and error paths.
  Compare residual/Jacobian values and accepted/rejected/refinement traces,
  with unchanged iteration/factorization counts on the baseline fixtures.
- [ ] Accumulate debug-validated changes before the next release build; run
  isolated evaluator and end-to-end stories together. Keep diagnostic Detailed
  and production Off measurements distinct. Accept measured warm gains only
  if cold preparation, Sparse and small controls remain within a documented
  noise band and correctness gates pass. Keep historical records intact.

Shared evaluator work belongs to `symbolic/View`, linked from
`src/symbolic/TODO.md`. AOT and large Dense runs are outside this investigation.

## 27. AOT Production Path: AtomView-Native Migration (2026-09-21, planned)

This is the next AOT architecture block after the pure-Lambdify baselines.
The target is a first-class `AtomView -> CodegenIR -> Rust/C/Zig AOT` route.
AtomView must not pass through `Atom -> Expr` merely because the selected
matrix backend or compiler still uses the historical AOT adapter. `ExprLegacy`
remains a separate, permanently exercised oracle and compatibility route.

The migration must advance four contracts together: prepared ownership,
fallible runtime errors, typed telemetry/logging, and correctness/performance
tests. No AOT backend is production-ready until all four contracts have a
matching gate.

Progress (2026-09-21): the first non-invasive slice adds a typed
`AtomAotPreparedPlan`, explicit Sparse/Banded layout metadata, fallible plan
validation and an Atom codegen-bridge accessor. This is preparation plumbing,
not yet a claim that the linked AOT runtime is AtomView-native.

Progress (2026-09-21, layout/dispatch slice): the common codegen model now
preserves `BandedValues { kl, ku, slots }` instead of erasing Banded into
`SparseValues`. Rust/C/Zig emitters retain the same contiguous callback ABI,
while AtomView AOT dispatch now selects the Atom codegen branch before
constructing the generic Expr `PreparedProblem`; Banded receives the explicit
layout metadata as well. Native band-slot packing, route markers and linked
runtime ownership remain open, so this is not yet the production exit gate.

Progress (2026-09-21, telemetry slice): added typed AOT `Off/Counters/Detailed`
telemetry with separate cold-stage, warm-callback and artifact-lifecycle
fields, and attached the handle to the prepared Atom plan. The schema is now
tested. Cold lifecycle events also have a typed `log::debug!` adapter; compiler,
link and warm-runtime integration remain open.

Progress (2026-09-21, native lifecycle dispatch slice): AtomView backend
selection now uses an Atom-derived manifest directly, and the selected bridge
stores the prepared Atom codegen payload instead of materializing its Expr
compatibility vectors. Registry keys and artifact manifests use the same
Atom-derived identity, and sparse structure metadata is reconstructed from
Atom entries for the solver handoff. The old `prepare_sparse_aot_problem` API
still remains an explicit compatibility adapter; this is therefore a native
canonical route plus compatibility bridge, not yet the final separate
`ExprLegacyAotAdapter`/`AtomViewAotAdapter` type split.

Progress (2026-09-21, route identity slice): the selected BVP AOT bridge now
exposes a typed `BvpAotPreparationRoute` marker and reports it through the
selection log and diagnostics (`AtomViewNative` versus `ExprLegacy`). Manifest
selection no longer resolves an AOT artifact at all for explicit
`LambdifyOnly`/`NumericOnly` policies. Native Atom payloads keep one sparse
source layout and apply Banded layout only for the requested matrix route,
avoiding cross-backend identity ambiguity. Debug contract tests cover native
Banded manifest construction and registry reuse by the manifest key.

### 27.1 Route and ownership contract

- [ ] Introduce a typed `AtomAotPreparedPlan` (name may change) owning residual
  atoms, sparse derivative atoms, Banded derivative atoms, ordered `Symbol`
  input schema, parameter schema, matrix layout, chunking policy, compiler
  profile and artifact identity. It must own the data required by both cold
  code generation and warm linked callbacks.
- [ ] Keep `ExprLegacyAotAdapter` and `AtomViewAotAdapter` as distinct internal
  routes. Do not expand `BvpPreparedSparseAotProblem` with an optional mixture
  of `Expr` and `Atom`; that would hide which representation is active.
- [x] Select the frontend and matrix route before constructing any generic
  Expr-based prepared problem. The AtomView branch must not eagerly build
  compatibility Expr runtime plans before dispatching to Atom codegen. The
  remaining invalid-layout compatibility fallback is logged and is not yet a
  production exit-gate path.
- [ ] Keep `Atom -> Expr` conversion only at explicit compatibility boundaries
  and instrument it with a typed conversion counter. The canonical AtomView
  AOT route must have zero such conversions after preparation.
- [ ] Make the prepared plan the owner of mesh/layout metadata, callbacks,
  generated output ordering, artifact fingerprint and linked runtime state.
  Numeric rebind, mesh/BC changes, matrix-policy changes and Jacobian-pattern
  changes must invalidate the appropriate owned resource explicitly.
- [ ] Separate cold `PreparedPlan`/artifact lifecycle from warm linked runtime
  state. A warm callback must not rebuild symbolic structure, regenerate source,
  rematerialize an artifact or silently select a different layout.

### 27.2 Atom-native code generation and matrix layouts

- [x] Extend the common codegen output model with an explicit Banded values
  layout, including `kl`, `ku`, slot ordering and row/column ownership. Do not
  represent Banded output as dense staging or as an Expr-based sparse fallback.
- [ ] Generate residual and Sparse Jacobian values directly from Atom views,
  preserving the existing fixed-CSC ordering and duplicate-coordinate rules.
- [ ] Generate Banded Jacobian values directly from Atom views, preserving the
  native band-slot contract and out-of-storage semantics. The current slice
  preserves the explicit band metadata and contiguous values, but native
  full-slot packing/out-of-storage behavior is still pending.
- [ ] Make Rust, C and Zig emitters consume the same Atom-derived IR and output
  manifest. ABI order must be explicit and identical: parameters followed by
  unknown/state inputs, then residual or Jacobian output buffers.
- [ ] Add a route marker to the prepared artifact and linked runtime so reports
  can distinguish `AtomView-native`, `ExprLegacy`, and compatibility fallback.
  A fallback must be visible and never silently labelled AtomView-native.

### 27.3 Typed errors and safe runtime boundary

- [ ] Add fallible `try_residual_into` and `try_jacobian_values_into` AOT
  boundaries. Replace runtime `assert!`, `assert_eq!`, `unwrap` and `expect`
  on user/input/artifact state with typed errors.
- [ ] Cover typed errors for missing or stale artifacts, wrong problem key,
  ABI/schema mismatch, parameter-count mismatch, output-shape mismatch,
  sparse/ Banded layout mismatch, link/load failure, non-finite callback output
  and invalidation during rebind.
- [ ] Preserve compatibility panic-style wrappers only outside the new typed
  path and document them as legacy adapters.
- [ ] Retain partial preparation/runtime diagnostics when a failure occurs:
  last completed stage, route, layout, artifact identity, retry/quarantine
  action and elapsed stage times must remain available in the error report.

### 27.4 AOT telemetry and logging

- [ ] Add one typed AOT telemetry mode shared by AtomView and ExprLegacy:
  `Off`, `Counters` and `Detailed`. `Off` must not allocate, start timers,
  format strings or update maps in the hot path.
- [ ] Split cold telemetry into validation, Atom discretization, symbolic
  structure/Jacobian preparation, Atom lowering, optimization/temp reuse,
  source emission, materialization, compiler build, link/load, publication,
  cache hit/miss, retry and quarantine.
- [ ] Split warm telemetry into parameter binding, residual requests,
  Jacobian requests, scalar evaluations, chunks/tasks, effective workers,
  callback elapsed time, output writes, copies, allocations where measurable,
  non-finite/error exits and runtime fallback selections.
- [ ] Keep callback counters semantically comparable with Lambdify: a residual
  request, Jacobian request, evaluator task and emitted scalar operation must
  be separate counters, not merged into one backend-dependent number.
- [ ] Replace primary AOT runtime diagnostics maps with fixed typed snapshots.
  A `HashMap<String, String>` may remain only as a compatibility/presentation
  projection outside the hot path.
- [ ] Add typed lifecycle log events for planned, source-emitted,
  materialized, build-started, build-succeeded/failed, published,
  linked/loaded, runtime-ready, retry and quarantine transitions. Include
  route, matrix layout, toolchain, artifact key and selected chunk policy.
- [ ] Aggregate worker-thread telemetry through the prepared runtime or fixed
  counters, not fragile TLS-only callback state. Logging and report rendering
  must happen outside timed callback scopes.

### 27.5 Correctness and story-test infrastructure

- [ ] Add a no-conversion gate proving AtomView AOT preparation and warm
  execution never call `atom_to_expr`; keep ExprLegacy as an independent oracle.
- [ ] Add componentwise residual and Jacobian parity at identical inputs and
  parameter bindings for ExprLegacy versus AtomView AOT.
- [ ] Add fixed-CSC and Banded-slot parity, including ordering, duplicate
  entries, bandwidth, output buffer writes and native storage values.
- [ ] Add prepared lifecycle tests for numeric rebind, mesh/BC mutation,
  Jacobian-pattern mutation, matrix-policy change, artifact replacement,
  stale-factor rejection and repeated warm solves.
- [ ] Add sequential/parallel/Auto and chunking parity tests. Record effective
  task count and selected policy; do not infer parallel execution from a
  configured flag alone.
- [ ] Add cold versus warm stories for small Dense control, production Sparse
  and production Banded workloads. Dense must remain a small correctness
  control, never a large-AOT performance target.
- [ ] Run the same stories for available Rust, C and Zig toolchains. Separate
  preparation/build/link time from warm residual/Jacobian/linear execution.
- [ ] Save printed AOT story results in the dedicated test-report files with
  canonical test name and timestamp. Preserve historical rows; append new
  dated records rather than overwriting earlier AOT evidence.
- [ ] Add failure-injection stories for compiler failure, partial artifact,
  stale publication, lock contention, link failure and quarantine/rebuild.
  Verify typed diagnostics and recovery without corrupting a valid artifact.

### 27.6 Production exit gate and order

1. Land the typed Atom plan and route marker without changing defaults.
2. Add Atom-native Sparse and Banded layouts and prove no Expr conversion.
3. Add fallible callbacks and lifecycle invalidation before performance claims.
4. Add cold/warm telemetry and logging, then validate `Off` overhead.
5. Run debug parity, failure-injection and lifecycle suites.
6. Run release stories with at least five warm samples and preserved historical
   baselines; compare stage timings, integer counters, allocations/copies and
   numerical decisions, not wall-clock alone.
7. Only after the evidence is stable, optimize lowering, temporary reuse,
   chunking and Auto break-even policy.

- [ ] Exit gate: AtomView AOT has no implicit Expr conversion on the canonical
  route and every fallback is explicit in the resolved plan.
- [ ] Exit gate: Sparse/Banded layout and numerical parity pass for all active
  toolchains and parameter rebind cases.
- [ ] Exit gate: typed errors cover preparation, artifact lifecycle and warm
  callback failures without process-aborting user-input paths.
- [ ] Exit gate: telemetry/logging schemas are comparable with Lambdify,
  disabled overhead is measured, and worker-thread aggregation is reliable.
- [ ] Exit gate: historical ExprLegacy records remain intact and no AOT
  performance claim is made without dated release evidence.
