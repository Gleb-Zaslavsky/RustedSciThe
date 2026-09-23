# TODO: Coordinated ExprLegacy to Prepared AtomView Migration

Priority update, 2026-09-23: pause further LSODE2/BVP evaluator migration while
investigating [View/Lambdify execution cost](View/TODO.md). Atom-derived
Jacobian expressions can evaluate more slowly despite fast preparation.
The target below is conditional on workload-specific correctness and measured
cold/warm performance, not a mandate to replace every Expr evaluator.

Execution priority updated on 2026-09-18: start with BVP Damped/Frozen using
the [local architecture review and checklist](../numerical/BVP_Damp/TODO.md).
The numbered workstreams below describe scope; the rollout section determines
execution order. LSODE2 follows the BVP pilot.

## Objective

Move solver-facing symbolic preparation and warm residual/Jacobian evaluation
from the legacy boxed-`Expr` callbacks to one reusable packed-Atom backend.

The migration must remove repeated `Expr <-> Atom` materialization, hot-path
matrix reconstruction, and mutex-protected Jacobian writes without changing:

- the public `Expr` API;
- BVP/IVP numerical algorithms or convergence behavior;
- the generated AOT ABI;
- solver-facing matrix semantics;
- existing task-document and builder syntax.

This is a coordinated migration, not a rewrite. `ExprLegacy` remains an
explicit compatibility and parity route until every solver family has passed
its own correctness and release-performance gates.

## Confirmed Starting Point

- [x] `View::AtomView` already is the required borrowed umbrella view. It is a
  `Copy` enum over packed `Num`, `Var`, `Fun`, `Pow`, `Mul`, and `Add` nodes.
  Adding another shallow `ExprView` would mostly duplicate `match &Expr` and is
  not, by itself, a performance improvement.
- [x] The View subsystem already provides packed parsing, normalization,
  differentiation, evaluation, printing, reusable workspaces, and Atom-native
  codegen lowering.
- [x] `PreparedSparseAtomSystem` converts `Expr` residuals once, records sparse
  variable-to-column candidates, and differentiates directly over packed
  atoms. It can also start from already packed atoms.
- [x] BVP and IVP expose explicit `ExprLegacy` and `AtomView` assembly choices.
  The migration therefore has an existing opt-in/rollback surface.
- [x] LSODE2 already has backend parity gates across Lambdify/AOT,
  ExprLegacy/AtomView, and dense/sparse/banded structures.
- [x] BVP_Damp already has symbolic Jacobian parity, endpoint correctness,
  sparse/banded comparison, AOT lifecycle, and production-sized story tests.
- [x] Nonlinear systems already demonstrate the desired prepared/bound
  lifecycle and mutex-free sequential/parallel `residual_into` and
  `jacobian_into` execution. Reuse its proven ideas, but do not couple solver
  modules to its concrete types. Its scalar evaluators still compile `Expr`;
  this is evidence for buffer/execution improvements, not for Atom runtime
  speed. Benchmark representation changes separately.

## Confirmed Technical Debt

- [ ] `symbolic_ivp::build_symbolic_jacobian(AtomView)` differentiates through
  `PreparedSparseAtomSystem`, then converts every nonzero `Atom` back to
  `Expr`, fills a dense `Vec<Vec<Expr>>`, and Lambdifies that tree again.
- [ ] IVP Lambdify callbacks clone parameter values under `RwLock`, allocate a
  flattened `time + parameters + state` vector, and allocate owned result
  vectors/matrices on every call.
- [x] BVP `install_atom_discretized_system` keeps residuals, variables, and
  metadata in the Atom-native discretized system. Expr compatibility data is
  materialized only by an explicitly requested legacy/compatibility accessor.
- [x] BVP AtomView still maintains an Expr compatibility cache for legacy
  consumers and the generated-backend bridge, but native Banded Lambdify no
  longer reads that cache at runtime.
- [x] BVP Dense/faer Lambdify now has an Atom-native callback branch that keeps
  packed residuals and sparse derivative atoms through callback compilation;
  the Expr conversion remains an explicit compatibility cache.
- [x] Split BVP callback implementations into `bvp::legacy_lambdify` and
  `bvp::atom_lambdify`; both use the common lock-free telemetry schema with
  independent streams.
- [x] Complete the no-round-trip path for native Banded Lambdify. The direct
  Banded owner compiles packed residual and sparse derivative atoms directly;
  the `Expr` payload remains only as an explicit compatibility cache.
- [ ] Replace the separate Dense/faer/Banded callback adapters with one typed
  prepared Atom runtime owner at the solver boundary.
- [ ] The legacy dense BVP and IVP Jacobian callbacks allocate a matrix and
  acquire one `Mutex` lock for each accepted nonzero value.
- [ ] The legacy faer sparse BVP callback pushes triplets through a shared
  `Mutex<Vec<Triplet<_>>>` and reconstructs the sparse matrix structure on
  every Jacobian call.
- [ ] `symbolic_functions2` clones symbolic matrices/vectors during callback
  construction, allocates flattened inputs per call, and uses the same
  per-nonzero mutex write pattern.
- [ ] The BVP banded path is already structurally more advanced than the dense
  and sparse legacy paths. It must be measured and adapted, not replaced
  blindly with a generic implementation that loses its native layout.
- [ ] `expr_to_atom` and `atom_to_expr` are compatibility conversions rather
  than a lossless universal representation contract. Some unsupported or
  oversized values can panic or be approximated, so the production prepared
  route needs typed conversion/preparation errors.

## Non-Goals

- [ ] Do not replace `Expr` with `Arc<ExprNode>`, arena IDs, or a global DAG
  without profiling evidence that packed Atom preparation still leaves a
  material bottleneck.
- [ ] Do not remove `symbolic_functions.rs`, `symbolic_functions2.rs`, or
  `symbolic_functions_BVP.rs` during the migration.
- [ ] Do not infer that parallel evaluation is always faster. Sequential must
  remain the baseline and small-workload fallback.
- [ ] Do not change LSODE2 method switching, BVP discretization, damping,
  linear solver selection, or convergence tolerances while changing the
  symbolic backend.
- [ ] Do not combine this work with a public API rename or task-document syntax
  migration.

## Target Architecture

The intended lifecycle is:

```text
Expr input / task document / direct Atom input
    -> PreparedAtomSystem
       - immutable residual atoms
       - ordered independent argument / parameter / unknown schema
       - symbolic sparse Jacobian values
       - fixed dense/sparse/banded destination layouts
       - sequential and parallel execution plans
       - codegen/AOT input view
    -> bind(parameter values)
    -> BoundAtomSystem
       - immutable parameter snapshot or validated binding handle
       - caller-owned/thread-local numeric scratch
    -> solver adapter
       - residual_into
       - jacobian_dense_into
       - jacobian_sparse_values_into
       - jacobian_banded_into
```

The canonical symbolic Jacobian should be stored as a fixed ordered list of
nonzero entries plus layout metadata. Dense, sparse, and banded forms are
adapters over that list; they must not independently repeat differentiation or
discover sparsity at runtime.

Suggested concepts, with final names to be chosen during implementation:

- `AtomInputSchema`: independent argument, ordered parameters, and ordered
  unknowns with validated, precomputed symbol slots.
- `PreparedAtomSystem`: immutable residual and symbolic Jacobian payload.
- `PreparedAtomJacobianLayout`: shape, ordered `(row, col)` entries, sparse
  pattern, optional bandwidth, and destination-specific write plans.
- `AtomExecutionPolicy`: `Sequential` and `Parallel { min_work }`.
- `BoundAtomSystem`: one validated parameter binding over shared immutable
  preparation.
- `AtomEvaluationScratch`: reusable flattened input and output workspace owned
  by a solve attempt or thread, never protected by one global hot-path mutex.

These types belong in `symbolic/View` or a sibling prepared-runtime module.
They must not live inside BVP_Damp, LSODE2, or Nonlinear_systems.

## Phase 0: Baseline And Consumer Inventory

- [ ] Enumerate every production caller of these legacy entry points before
  changing them:
  - `Jacobian::jacobian_generate_IVP_DMatrix`;
  - `symbolic_functions2` parameterized residual/Jacobian constructors;
  - BVP dense, sprs, faer sparse, and banded Lambdify constructors;
  - LSODE2 dense/sparse/banded symbolic preparation;
  - generated/AOT task construction that still expects `Expr` entries.
- [ ] Record the inventory as a migration matrix with columns: consumer,
  symbolic assembly, callback evaluator, matrix layout, parameter binding,
  AOT handoff, parity gate, and migration status.
- [ ] Add counters available in tests/benchmarks for:
  - `Expr -> Atom` conversions;
  - `Atom -> Expr` conversions;
  - symbolic differentiation count;
  - warm callback allocations and allocated bytes;
  - sparse structure rebuilds;
  - mutex acquisitions in residual/Jacobian callbacks.
- [ ] Establish release baselines before implementation. Separate cold
  discretization/parsing/differentiation/compilation from warm residual,
  Jacobian, linear solve, and total solve time.
- [ ] Use representative workloads rather than synthetic matrices alone:
  LSODE2 combustion and three-body, BVP combustion at 1000/3000 points,
  endpoint BVP banded cases, and one small problem proving overhead behavior.

## Phase 1: Shared Prepared Atom Runtime

- [ ] Generalize `PreparedSparseAtomSystem` into, or compose it with, an
  immutable `PreparedAtomSystem` that owns both residual atoms and Jacobian
  atoms. Preserve the current small focused Jacobian builder API.
- [ ] Accept both `&[Expr]` and `&[Atom]`. The Expr constructor must perform
  exactly one conversion per residual; the Atom constructor must perform no
  compatibility conversion.
- [ ] Validate dimensions, duplicate names, parameter/unknown collisions,
  undeclared symbols, bandwidth, and destination shape before compiling any
  callbacks.
- [ ] Replace conversion `expect`/panic paths with a typed preparation error
  carrying residual/Jacobian position and the unsupported construct.
- [ ] Store a stable ordered nonzero layout. Structural zero detection and
  `(row, col)` ordering must not depend on Rayon scheduling.
- [ ] Prepare variable and parameter lookup once. Symbol-name maps, string
  creation, and symbolic traversal must not occur during warm callbacks.
- [ ] Compile residual and Jacobian Atom evaluators once. Investigate a batch
  evaluator only after measuring whether one evaluator per expression is a
  meaningful dispatch cost.
- [ ] Define `Send + Sync` guarantees explicitly. Immutable preparation may be
  shared; output and scratch buffers belong to each caller/solve/thread.
- [ ] Define parameter binding semantics explicitly:
  - immutable bound snapshots are the canonical concurrency-safe path;
  - mutable shared handles may remain compatibility adapters;
  - a failed rebind must leave the previous valid binding intact;
  - parameter-only changes must never rebuild symbolic derivatives.
- [ ] Feed the same prepared Atom payload to Lambdify evaluation and
  `CodegenIR_atom`. Do not maintain separate symbolic Jacobians for AOT and
  Lambdify.

## Phase 2: Mutex-Free Runtime Evaluation

- [ ] Implement allocation-reusing `residual_into` and Jacobian-value
  evaluation into caller-owned buffers.
- [ ] Make sequential execution the reference implementation and default.
- [ ] Add explicit `Parallel { min_work }` execution. Reuse the Nonlinear
  systems policy semantics where practical so users do not learn unrelated
  threshold concepts for each solver family.
- [ ] Partition work during preparation, not during every callback.
- [ ] Give each parallel worker an exclusive output range. No worker may lock
  a matrix or append to one shared triplet vector.
- [ ] For dense nalgebra output, precompute destination offsets in
  column-major storage and write disjoint ranges or evaluate into an ordered
  values buffer followed by a measured layout adapter.
- [ ] For faer sparse output, prepare symbolic/index structure once and update
  only the numeric values array. Do not rebuild triplets in the warm path.
- [ ] For banded output, write directly into the existing native band storage
  and preserve the proven BVP_Damp bordered/banded linear-solver contract.
- [ ] Threshold numerical values only if the existing backend contract
  requires it. Structural sparsity must come from symbolic preparation, not
  from dropping small runtime values and changing the matrix pattern.
- [ ] Reuse flattened input scratch after the first call. If copying
  `argument + parameters + unknowns` is still material at production sizes,
  benchmark a split-input evaluator before changing the evaluator ABI.
- [ ] Keep solver-level callback counters distinct from generated chunk/job
  counters. Lambdify and AOT diagnostics must count at the same abstraction
  level.

## Phase 3: LSODE2 And Shared IVP Follow-Up

LSODE2 follows the BVP_Damp pilot. Its symbolic preparation is already
centralized in `symbolic_ivp`, its backend configuration is explicit, and its
parity/story infrastructure covers all relevant matrix structures.

- [ ] Add a direct prepared-Atom Lambdify branch to `symbolic_ivp`; do not
  route it through `Vec<Vec<Expr>>` or `Expr::lambdify_*`.
- [ ] Preserve the exact flattened schema `time, parameters..., states...`
  currently shared with AOT.
- [ ] Replace per-call parameter `DVector` cloning and argument-vector
  allocation with validated binding plus reusable scratch.
- [ ] Expose residual-only preparation without constructing a dense symbolic
  Jacobian for native sparse/banded routes.
- [ ] Connect dense, sparse, and banded LSODE2 adapters to the same prepared
  nonzero values and fixed layout.
- [ ] Preserve `ExprLegacy` as an explicit builder/config option and make the
  direct Atom path opt-in until all gates pass.
- [ ] Do not modify LSODA/LSODE switching or retry logic in the same changeset.
- [ ] Confirm that AOT materialization consumes the same prepared Atom graph;
  keep the current generated ABI and artifact lifecycle unchanged initially.

### LSODE2 Gates

- [ ] Reuse `lsode2_backend_parity_checklist_lambdify_and_aot_exprlegacy_and_atomview`.
- [ ] Reuse dense/sparse/banded AtomView end-to-end correctness tests.
- [ ] Reuse Lambdify versus prelinked-AOT elementwise equivalence tests.
- [ ] Add direct `ExprLegacy` versus prepared-Atom componentwise residual and
  Jacobian parity at several states, including parameters and time dependence.
- [ ] Add fixed sparse-pattern and inferred-bandwidth parity.
- [ ] Add warm callback allocation/conversion assertions: no Atom-to-Expr
  conversion, no structure rebuild, and no mutex acquisition.
- [ ] Require identical solver-level residual/Jacobian/linear counts,
  accepted/rejected steps, method-switch trace, and final solution within the
  existing parity tolerances.
- [ ] Add a release story showing cold preparation and warm callback/solve
  stages separately. Parallel promotion requires evidence on a genuinely
  large callback workload; three-body remains a small-workload Whole/Sequential
  control.

## Phase 4: BVP_Damp Migration (First Production Pilot)

Use the [BVP-local plan](../numerical/BVP_Damp/TODO.md) for factorization
lifetime, typed matrix adapters, pure numerical assembly and telemetry work
that must accompany the symbolic migration. Banded already has mutex-free
diagonal evaluation; preserve it when introducing reusable output buffers.

- [ ] Reuse `DiscretizedBvpAtomSystem` as the Atom-native source of residuals,
  variables, boundary metadata, bounds, and tolerances.
- [x] The Atom runtime path no longer requires
  `install_atom_discretized_system` to materialize residuals and variables back
  into `Expr`; the compatibility view is explicit and lazy.
- [x] Native AtomView Jacobian entries remain `Atom` after differentiation and
  are consumed directly by AtomView Lambdify callback preparation.
- [ ] Compile one shared residual/Jacobian value plan, then adapt it to dense,
  faer sparse, and native banded storage.
- [ ] Remove per-nonzero dense matrix locking.
- [ ] Remove the shared sparse triplet mutex and per-call sparse pattern
  construction.
- [ ] Audit the current banded evaluator before changing it. Preserve any
  existing fixed-layout or allocation reuse that is already better than the
  generic dense/sparse legacy routes.
- [ ] Preserve Damped and Frozen public builder options, task-parser mappings,
  derivative scheme, grid refinement, bounds, and postprocessing behavior.
- [ ] Keep AOT backend/task configuration sourced from the same prepared
  symbolic payload and preserve BuildIfMissing/RequirePrebuilt behavior.

### BVP_Damp Gates

- [ ] Reuse `symbolic_assembly_backends_match_two_point_jacobian_on_small_sparse_bundle`.
- [ ] Reuse AtomView two-point and oscillator end-to-end tests for Lambdify
  and AOT.
- [ ] Reuse combustion sparse/banded and 1000/3000-point story tests.
- [ ] Add direct prepared-Atom versus current Atom-to-Expr Lambdify callback
  parity before deleting either internal route.
- [ ] Compare residuals, Jacobian nonzero values, sparse indices, band layout,
  Newton iteration history, refinement decisions, final mesh, and solution.
- [ ] Add warm allocation/conversion/mutex assertions analogous to LSODE2.
- [ ] Re-run stage timings (`residual_ms`, `jacobian_ms`, `linear_ms`, total)
  with identical numerical work and multi-run summaries.
- [ ] Do not promote a default based only on AOT stories; the new direct Atom
   Lambdify path needs its own production-size evidence.

### BVP status audit (2026-09-22)

The dated status below is the current BVP-specific reading of the migration
checklist. The Phase 4 checkboxes above remain the historical work plan; this
section prevents completed AtomView work from being mistaken for open work,
without claiming that the whole BVP runtime has already been replaced.

Completed or demonstrated in the current BVP pilot:

- [x] `bvp::atom_lambdify` has native AtomView callback paths for Dense, faer
  sparse, and Banded Lambdify. The native callback path does not read the
  compatibility `Expr` cache at runtime.
- [x] `Sequential`, explicit `Parallel`, and `Auto` execution policies exist
  for AtomView callbacks, with debug correctness coverage for residuals,
  Jacobians, sparse layouts, and Banded layouts.
- [x] BVP symbolic/runtime responsibilities are separated into explicit
  `legacy_symbolic`, `legacy_lambdify`, `atom_lambdify`, and direct Banded
  modules; the old public module paths remain compatibility facades.
- [x] Parameter rebinding, fixed sparse-pattern checks, compact Banded slot
  checks, and callback-stage telemetry are covered by the current BVP test
  corpus. Release measurements exist for the main Lambdify routes, but must be
  refreshed after the current module split before being treated as final.
- [x] The legacy Mutex/Expr callback implementation remains available as a
  separate numerical oracle. It is intentionally retained for parity and
  regression localization, not used as the AtomView hot path.

Still open or only partially complete for BVP production readiness:

- [ ] Finish one typed prepared Atom runtime owner at the solver boundary for
  callbacks, mesh/layout, Jacobian value plans, and numeric factors. The
  current callbacks are separated, but Dense/faer/Banded are not yet owned by
  one complete `PreparedPlan` runtime.
- [ ] Audit the remaining explicit Expr materialization at compatibility
  boundaries, especially the retained `Vec<Expr>` AOT adapter. Native BVP
  Lambdify assembly and callbacks are already Atom-native; this item is not a
  claim that ordinary AtomView Lambdify still performs an implicit round-trip.
- [ ] Complete the invalidation matrix for parameters, mesh, boundary
  conditions, state values, solver policy, Jacobian pattern, and factor
  ownership. Prove that rebind/refinement/policy changes cannot reuse a stale
  callback or factor.
- [ ] Extend componentwise parity beyond final solutions: accepted/rejected
  Newton traces, damping trials, refinement decisions, callback values, fixed
  CSC ordering, compact Banded slots, and final mesh for ExprLegacy versus
  AtomView and Sequential/Parallel/Auto.
- [ ] Finish the fallible boundary for symbolic conversion, callback shape/
  value failures, and linear-runtime failures. Compatibility panic wrappers may
  remain, but new BVP user paths must return typed errors with partial
  diagnostics.
- [ ] Close the warm-path allocation audit: residual/Jacobian input and output
  buffers, triplet/index construction, conversions, copies, worker chunks, and
  factor storage bytes. Dense remains a small-problem control; optimization
  priority is AtomView Sparse and Banded.
- [ ] Refresh the release baseline after the current refactor using identical
  numerical work and repetitions. Record cold preparation, warm callbacks,
  factorization/RHS solve, integer counters, and per-stage telemetry in the BVP
  story files before changing the hot path again.
- [ ] Keep AOT lifecycle and AtomView-native AOT generation as a separate gate;
  the BVP Lambdify checklist must not mark AOT production-ready merely because
  native Lambdify callbacks are correct.

## Phase 5: Legacy API Isolation And Compatibility

- [x] Physically separate the ExprLegacy symbolic-preparation helpers from
  the main BVP implementation: smart/optimized/full parallel differentiation
  and bandwidth discovery now live in `bvp::legacy_symbolic`, while public
  `Jacobian` methods remain compatibility wrappers. The larger
  discretization/generation orchestrator is intentionally the next boundary.
- [ ] Move the old Mutex-based callback implementations into clearly named,
  physically separate legacy modules without changing public paths initially.
  Do not leave old and new execution engines interleaved in the same large
  implementation files.
- [ ] Preserve the known-correct legacy implementations permanently as a
  numerical reference/oracle. Do not delete them after the prepared Atom
  backend becomes the production default: they are required for componentwise
  residual/Jacobian parity, solver regression localization, and validation of
  future optimizations.
- [ ] Keep legacy reference modules buildable and covered by a compact
  correctness suite. They may be hidden from normal high-level UX or selected
  through an explicit diagnostic/compatibility option, but they must not rot
  behind permanently disabled code.
- [ ] Keep the reference role explicit: legacy code is not the preferred hot
  path and must not leak Mutexes, allocations, or fallback behavior into the
  prepared Atom implementation.
- [ ] Turn compatible methods in `symbolic_functions.rs`,
  `symbolic_functions2.rs`, and `symbolic_functions_BVP.rs` into thin wrappers
  over the prepared runtime where return types and semantics match.
- [ ] Preserve old owned-return closures as adapters around caller-owned
  `*_into` methods. Allocation in a compatibility adapter must be visible and
  must not contaminate the canonical runtime.
- [ ] Keep truly incompatible legacy behavior isolated rather than silently
  approximating it with a new route.
- [ ] Add deprecation notes only after all in-crate production consumers have
  migrated and guides show the replacement API.
- [ ] Do not remove `ExprLegacy` after default promotion. Keep it as an
  explicit diagnostic/parity route; only its public visibility and support
  level may be reconsidered in a separate compatibility decision.

## Phase 6: AOT And Artifact Identity Alignment

- [ ] Make Lambdify and AOT consume the same canonical residual and Jacobian
  ordering from `PreparedAtomSystem`.
- [ ] Include symbolic representation/layout version in artifact identity so
  stale pre-migration artifacts cannot be mistaken for compatible outputs.
- [ ] Keep numeric parameter values out of artifact identity; include ordered
  parameter schema and every compile/layout option that changes generated code.
- [ ] Verify Whole/Chunked codegen uses the same fixed nonzero ordering as the
  direct Atom evaluator.
- [ ] Preserve the existing row-major generated ABI until a separate measured
  migration justifies changing it.
- [ ] Re-run lifecycle, lock, retry, failed-build quarantine, manifest/hash,
  and RequirePrebuilt tests after the common prepared input is installed.

## Correctness Gates Missing From Existing Coverage

Existing solver parity is strong and should be reused rather than duplicated.
The migration specifically needs the following focused additions:

- [ ] Round-trip-free evaluation parity: ExprLegacy callback versus direct
  Atom callback for every supported operation and function.
- [ ] Componentwise residual and Jacobian parity, not only final-solution
  parity.
- [ ] Dense, sparse, and banded adapters reconstructed from one prepared value
  list must represent the same matrix.
- [ ] Stable nonzero ordering and deterministic parallel results across
  repeated and concurrent calls.
- [ ] Parameter order, repeated binding, failed binding rollback, and
  independent concurrent bindings.
- [ ] Structural-zero behavior when a numerical value crosses zero.
- [ ] Empty rows/columns, diagonal-only, asymmetric bandwidth, endpoint rows,
  and boundary-condition entries.
- [ ] Non-finite evaluation must return the same typed failure in Sequential,
  Parallel, Lambdify, and AOT routes.
- [ ] Unsupported conversion/evaluation constructs must return typed errors,
  never panic in a user-facing preparation call.
- [ ] Direct Atom and Expr inputs representing the same equations must produce
  identical schemas, layouts, callbacks, and artifact identities where the
  representations are semantically equivalent.

## Telemetry Must Evolve With Production Code

Telemetry is part of the production architecture, not a cleanup task after the
new backend is complete. Every migration changeset that introduces a new
preparation, binding, callback, layout-adaptation, parallel, or AOT stage must
add or update its diagnostics in the same change.

- [ ] Define one shared symbolic-backend telemetry schema used by LSODE2,
  BVP_Damp, Lambdify, and AOT adapters where the stages have the same meaning.
- [ ] Keep solver-level counters semantically stable across backends:
  `residual_calls`, `jacobian_calls`, and `linear_calls` count solver requests,
  not generated chunks, worker jobs, scalar expressions, or FFI calls.
- [ ] Report backend-internal work separately: residual/Jacobian evaluator
  jobs, chunks, evaluated nonzeros, layout copies, FFI calls, and fallback
  selections must never be mixed into solver-level callback counters.
- [ ] Extend cold-stage telemetry together with the prepared runtime:
  validation, Expr-to-Atom conversion, symbolic differentiation, sparsity and
  bandwidth discovery, evaluator compilation, execution-plan partitioning,
  code generation, artifact materialization, build, link, and load.
- [ ] Extend warm-stage telemetry together with each adapter: parameter/input
  binding, residual evaluation, Jacobian-value evaluation, layout adaptation,
  sparse/banded assembly, and total callback duration.
- [ ] Publish the resolved execution plan in diagnostics: symbolic backend,
  execution policy, active Sequential/Parallel decision, work threshold,
  chunk count, matrix layout, nonzero count, inferred bandwidth, AOT toolchain,
  and artifact lifecycle action.
- [ ] Preserve partial telemetry when preparation, compilation, loading, or
  evaluation fails. Diagnostics must identify the last completed stage and
  retain elapsed times and resolved-plan data collected before the failure.
- [ ] Add explicit counters for migration invariants: Expr-to-Atom and
  Atom-to-Expr conversions, symbolic differentiation, sparse-structure
  rebuilds, output-buffer allocations, and Jacobian mutex acquisitions.
- [ ] Make telemetry opt-in or low-overhead as appropriate, but never compile
  away the correctness counters required by mandatory parity tests without an
  explicit test configuration.
- [ ] Benchmark the price of telemetry for small and production-sized systems.
  Report disabled versus enabled timings and allocations; do not infer overhead
  from one noisy sub-millisecond run.
- [ ] Add schema/semantics regression tests so a refactor cannot silently
  redefine a counter, merge unlike stages, or make Lambdify and AOT reports
  incomparable.
- [ ] Update story-table printers and English/Russian diagnostics guides when
  telemetry fields change. Mark old conclusions as historical/superseded when
  their counters used a different abstraction level.
- [ ] Treat missing diagnostics for a newly introduced production path as an
  incomplete migration item even when numerical correctness tests pass.

## Performance And Acceptance Policy

- [ ] Measure before and after in the same release binary, on the same machine,
  with preparation outside warm callback loops and at least five repetitions
  for story-level conclusions.
- [ ] Report cold stages separately: discretization/parsing, Expr-to-Atom,
  differentiation, evaluator compilation, AOT materialization/build/link.
- [ ] Report warm stages separately: residual, Jacobian, linear solve, total,
  callback counts, allocations/bytes, conversions, and mutex acquisitions.
- [ ] Treat small-task regressions and large-task wins separately. Parallel
  execution must fall back to Sequential below measured work thresholds.
- [ ] Required warm-path invariants for the canonical prepared Atom route:
  - zero Atom-to-Expr conversions;
  - zero symbolic differentiation;
  - zero sparse-structure rebuilds;
  - zero shared Jacobian-output mutex acquisitions;
  - no heap allocation after scratch warmup for caller-owned `*_into` calls,
    unless a documented external matrix API forces one.
- [ ] A solver default may change only when correctness gates are mandatory
  green and release stories show neutral-or-better end-to-end behavior on both
  a small control and representative production workloads.
- [ ] Performance stories remain advisory; parity tests remain mandatory. A
  noisy timing failure must not make the correctness suite flaky.

## Rollout Order

- [ ] Step 1: BVP baseline counters/benchmarks, factorization lifetime and common
  prepared callback contracts, following the local BVP plan.
- [ ] Step 2: BVP_Damp direct-Atom residual and fixed-layout sparse/banded
  Jacobian callbacks after isolated buffer/execution changes.
- [ ] Step 3: BVP_Damp parity and release stories; no default change yet.
- [ ] Step 4: reuse the shared preparation/runtime in direct-Atom IVP callbacks.
- [ ] Step 5: LSODE2 parity and release stories; no default change yet.
- [ ] Step 6: redirect compatible legacy APIs to prepared adapters and isolate
  the original implementations.
- [ ] Step 7: align AOT task construction and artifact identity with the common
  prepared payload.
- [ ] Step 8: consider per-solver AtomView default promotion independently.
- [ ] Step 9: update English/Russian guides and examples, marking historical
  story conclusions as superseded where methodology changed.

## First Symbolic Implementation Slice

The first symbolic slice follows the BVP baseline and ownership work described
in its local plan:

- [ ] Add `PreparedAtomSystem` composition around residual atoms,
  `PreparedSparseAtomSystem`, ordered input schema, and fixed nonzero layout.
- [ ] Add Sequential `residual_into` and `jacobian_values_into` with typed
  errors and reusable caller-owned buffers.
- [ ] Add unit parity against Expr evaluation and the current Atom-to-Expr
  Jacobian route.
- [ ] Add conversion/allocation counters proving preparation converts once and
  warm callbacks do not round-trip.
- [ ] Integrate the BVP_Damp Lambdify AtomView branch first.
- [ ] Run existing BVP symbolic/backend parity and the new direct-evaluator
  componentwise gates before migrating LSODE2.

Keep evaluator representation, parallel partitioning, linear-factor caching
and compatibility wrapper changes independently reviewable and measurable.
Default promotion remains a separate evidence-based decision.

## AtomView Numeric Evaluator Follow-up (2026-09-20, analysis only)

The concrete investigation and acceptance checklist is in
`../numerical/BVP_Damp/TODO.md`, section 26. The 23:33 Banded report uses parallel
execution on both frontends, but AtomView still has slower warm callbacks.
Its evaluator timer includes scheduling and slot writes, so scalar tape cost
must be isolated before choosing a representation change.

- [ ] Measure real BVP tape/opcode distributions, including negative powers
  introduced by Atom division, and compare identical-state scalar evaluation.
- [ ] Evaluate typed builtin/power specializations and trivial-entry fast paths
  with floating-point/domain parity gates and explicit error semantics.
- [ ] Evaluate batch-owned scratch and high-water reuse to amortize per-scalar
  TLS, arity checks and resizing; retain safe concurrent and reentrant use.
- [ ] Keep shared evaluator changes independent from Banded chunk partitioning;
  validate existing View callers and Sparse before promoting a fast path.
  Preserve general/custom-function and legacy implementations for comparison.

- [x] Add the first layout-aware Auto dispatch slice for direct Banded
  callbacks (2026-09-20, debug): long single diagonals expose coarse evaluator
  tasks instead of being reported as parallel while scheduling one Rayon job.
  Record the effective task count in typed direct telemetry. This is a
  correctness/observability gate only; release break-even tuning remains open.
