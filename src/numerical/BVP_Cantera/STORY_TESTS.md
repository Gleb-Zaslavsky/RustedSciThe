# BVP_Cantera Story Tests

## 2026-11-05: Initial release matrix and nonlinear perf-gate smoke

The initial full Release matrix passed all six stages. Cantera tests, including
ignored AOT and large-combustion cases, reported 53 passed, 0 failed, 0 ignored;
BVP_api facade tests reported 29 passed, 0 failed. Python binding compile, the
Rust facade example, the 12-route benchmark matrix and AOT Expr/Atom layout
parity also passed. This was superseded later the same day by the expanded
eight-stage matrix recorded below. Summary and logs:
`test_reports/BVP_Cantera_release_manual/20261105T055624Z/`.

The earlier same-day runs passed 48 and then 51 Cantera tests; the latest suite
also includes the symbolic Jacobian error-source and 18k workspace-capacity
regressions.

The 12-route release benchmark covers ExprLegacy/AtomView, Lambdify/AOT and
Dense/Sparse/BandedAuto on its exact trapezoid-decay fixture. All routes passed
with maximum profile error `4.441e-15`; each used one Newton iteration, two
residual evaluations, one Jacobian/factorization and one linear solve. The
single-run solve range was `0.062-12.939 ms`: Dense was about `12.7-12.9 ms`,
Sparse `0.137-0.233 ms`, and BandedAuto `0.062-0.082 ms`. This is a compact
backend/reference measurement, not a stable performance baseline. The report is
`test_reports/BVP_Cantera_release/release/bvp_cantera_workloads.md`.

For the gate workload, replaced the former independent-decay toy with the
nonlinear six-state combustion-like system also used by the BVP_sci workload
matrix, on 3000 fixed intervals (18000 reduced unknowns), using its tolerances
and bounds. Eight stable
IDs cover ExprLegacy/AtomView x Lambdify/AOT x Sparse/BandedAuto. Every solve
checks profile shape and finiteness, all endpoint values, bounds and residual.
All eight routes passed a Release pre-baseline smoke with one warmup and three
samples. Each batch executes four fresh solves. After adding a backward-residual
check to native Sparse/Banded solves and hoisting its matrix infinity norm out
of the per-RHS loop, the measured batch wall medians were:

| frontend | layout | execution | batch wall median (ms) |
|---|---|---|---:|
| ExprLegacy | Sparse | Lambdify | 97.385 |
| AtomView | Sparse | Lambdify | 95.811 |
| ExprLegacy | BandedAuto | Lambdify | 81.167 |
| AtomView | BandedAuto | Lambdify | 81.193 |
| ExprLegacy | Sparse | AOT | 95.494 |
| AtomView | Sparse | AOT | 96.739 |
| ExprLegacy | BandedAuto | AOT | 79.443 |
| AtomView | BandedAuto | AOT | 84.128 |

All runs reported 3 iterations/solve, 6 residual calls/solve, one Jacobian and
factorization/solve, residual `7.03734e-10`, exact endpoint values and no bound
violation. Native Sparse/Banded now also reject non-finite solutions and verify
the backward residual; this closes a concrete Faer Sparse case where a
singular system with an incompatible RHS had previously returned `Ok`. The
typed failure retains the partial solve report and has regression coverage.
The dedicated Debug lifecycle regression also forces a singular refreshed
Jacobian after one accepted Newton update on Dense, Sparse and Banded. It
confirms that each typed failure exposes the last accepted state and reports
only the factorizations/linear solves completed by that backend.
An additional Off/Counters/Timings parity test forces rejected damping trials,
an accepted update and iteration-limit failure; all modes preserve the same
state and iteration outcome, counted modes agree on work, and Off remains
empty.
The perf-gate run was intentionally a 3-sample dirty-source smoke; it is not a
baseline or a regression verdict. The approved `record` and `compare` pass
remains after the source is committed cleanly. Scenario IDs are distinct from
the former 1024-interval decay fixture, so no old references can be selected
for this combustion workload.

## 2026-11-05: Isolated Dense Newton core pilot

Created a separate callback-driven fixed-mesh Newton core so Cantera iteration
semantics can be tested without invoking the production Damped solve loop. The
first implementation is intentionally narrow: one packed residual/Jacobian
interface and nalgebra Dense LU. It includes weighted RMS correction scaling,
bounded updates, a fixed `sqrt(2)` damping reduction, Jacobian reuse, and typed
partial reports on failures.

This is an initial implementation checkpoint, not yet a claim of full Cantera
parity or combustion robustness. The corresponding Debug tests establish
basic linear convergence, bound preservation, telemetry-off behavior and
typed partial failure reports. Large combustion fixtures and exact reference
transition tests remain open in `TODO.md`.

## 2026-11-05: First large fixed-grid combustion gate

Added an ignored Debug gate using the shared 26-state, 33-column combustion
fixture from `BVP_Damp/tests/large_native_regressions.rs`. The standalone path
prepares callbacks through `ExprLegacy + Lambdify`, then runs its own Dense-LU
Newton loop; it does not invoke the Damped solve loop. The observed run
converged in 4 iterations with weighted correction `0.3440715742` and final
residual norm `2.812157626e-4`. Counters were 15 residual evaluations, 2
Jacobian evaluations/factorizations, 14 linear solves, 10 damping trials,
7 rejected trials and 3 accepted steps. Bounds and finiteness checks passed.

Also ported the six-state nonlinear combustion fixture used by the BVP_sci
cross-solver story, at 64 and 128 nodes. Both converged in 5 iterations with
one Jacobian evaluation/factorization and 10 linear solves. Final residual
norms were `1.221808034e-7` and `6.111295419e-8`; both profiles were finite and
within configured bounds.

An attempted profile comparison against the Damped facade could not be made
for the 26-state fixture: its `solve()` returned `Ok(())`, but
`solution_view()` had no published profile. This diagnostic attempt is not a
Cantera parity gate. Cantera parity concerns the Newton algorithm and its
transitions; cross-solver node counts and pointwise profile equality with
BVP_sci are not expected. Further stress cases and exact Cantera
source-transition parity remain open. These Debug runs make no performance or
production-stability claim.

## 2026-11-05: MultiNewton norm, age and failure-path parity

Compared the loop against the pinned [Cantera v3.1.0 `MultiNewton.cpp`](https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/MultiNewton.cpp).
This exposed a real norm mismatch: Cantera uses one error weight per physical
component, `atol + rtol * mean_mesh(abs(state_component))`, then returns the
square root of the sum of all scaled correction squares divided by the full
packed state size. The pilot had instead used each node's local state value.
`weighted_rms_uses_component_mean_magnitude_over_the_mesh` now pins the reference
formula.

The solver no longer makes an extra residual callback solely to populate the
final report. `residual_norm` is taken from the accepted trial already evaluated
by the Newton loop, so telemetry counts actual algorithm work. Callback timings
now expose residual and Jacobian subtotals as well as their aggregate. Logs
include Jacobian refreshes, each damping trial's bound factor/alpha/norms,
accepted steps, refresh retries, bounds-blocked exits and nonconvergence; all
remain behind the shared opt-in `SolverLogLevel` scope.

The Cantera reference refreshes `MultiJac` using a residual evaluation as part
of its finite-difference Jacobian assembly. This pilot instead receives an
already assembled symbolic Jacobian callback, so refresh intentionally does
not perform that reference-specific residual pre-evaluation. Its counters are
exact for the explicit ExprLegacy callbacks and Dense solves, but are not
expected to equal Cantera's internal finite-difference callback counts.

Added reference-contract tests for the strict `jacobian_age > max_age` refresh
boundary, typed fraction-to-boundary failure below `1e-10`, and one stale-
Jacobian damping-failure refresh. The retry test fixes the diagnostic counts at
18 residual evaluations, 18 linear solves and 15 damping trials over three
iterations. The full four-refresh retry limit and broader transition matrix
are now also pinned: the synthetic failure gate allows four refreshes, then
returns typed nonconvergence on iteration 10 with the last accepted state
`x=5`, 5 Jacobian evaluations, 50 residual evaluations, 50 linear solves and
40 damping trials (35 rejected). Broader edge-case transition coverage remains
open.

After the norm correction, both ignored combustion gates still pass. The
26-state fixture reports 4 iterations, correction `0.3444674060`, residual
`2.812157626e-4`, 14 residual evaluations, 2 Jacobian evaluations and 14 linear
solves. The six-state fixture reports 5 iterations at both 64 and 128 nodes;
residuals are `1.221808034e-7` and `6.111295419e-8`, with 10 residual
evaluations, 1 Jacobian evaluation and 10 linear solves per run. These are
Debug correctness observations, not a performance comparison or complete
Cantera parity verdict.

## 2026-11-05: KiThe combustion-failure workload gates

Added an explicit gate for the exact six-state stiff system from KiThe's
`rst_grid_refinement_diagnostics.rs`: six RHS equations, endpoint conditions,
initial guess, bounds and `1e-8` tolerances are retained. The AtomView adaptive
Damped route fails while preparing its Jacobian; the standalone Cantera
ExprLegacy/Dense fixed-mesh core converges on 33 nodes in one accepted Newton
update. It reports residual `3.367003941e-13`, 2 residual calls, 1 Jacobian
evaluation/factorization, 2 linear solves and 1 accepted step. Timings were
enabled for this diagnostic gate; observed callback total was about `1.3 ms`
(residual `0.68 ms`, Jacobian `0.63 ms`) and factorization about `33 ms` in
Debug. Those timings are diagnostic only and are not a performance claim.

The 26-state KiThe combustion reproducer is already covered by the fixed-grid
gate above. Its independent BVP_sci control succeeds from the 33-node seed and
publishes 78 nodes. BVP_sci and the Cantera core are different numerical
methods, so matching node counts or pointwise profiles is neither expected nor
an acceptance criterion. The relevant gates are that the Cantera core follows
the pinned Cantera Newton algorithm and converges acceptably on these same
problem fixtures. The separate 6-state BVP_sci cross-solver fixture at 64/128
nodes is a different workload, not a substitute for the KiThe cases.

Verification: `cargo test --lib numerical::BVP_Cantera:: --
--include-ignored --nocapture --test-threads=1` passed all 15 tests (12 core
contract tests and 3 workload gates). This closes the immediate requirement to
run Cantera numerical gates on the identified KiThe fixtures. Exact Cantera
transition parity, broader combustion coverage and Release/performance gates
remain open.

## 2026-11-05: Full-grid mapping and failure-transition corrections

The KiThe/BVP-derived symbolic adapter previously assumed that each packed
unknown's component could be recovered with `packed_index % component_count`.
That is wrong when boundary conditions eliminate different endpoints for
different components. The adapter now builds the reduced Newton vector in
node-major callback order while preserving each component's increasing order
of free mesh values, and carries an explicit mapping back to the full grid.
The weighted norm includes prescribed endpoint magnitudes in its per-component
mean and divides by the full state size; bounds still apply only to free
unknowns. A nonconstant mixed-left/right fixture pins values, positions,
endpoint data and the resulting norm.

Bounds-blocked damping status now follows Cantera's retry state machine: a
stale reused Jacobian may be refreshed within the same retry budget as an
exhausted line search, and a terminal block remains a typed
`BoundsBlocked` failure with the last accepted state. A separate test pins the
refresh, accepted-state and callback-counter behavior. The accepted unit
correction boundary is also explicitly tested: `s1 == 1` converges when the
trial is accepted, matching Cantera.

An adapter-level mixed-endpoint test now solves `y'=1`, `z'=2` with `y(0)=1`
and `z(1)=2`, checking the reduced node-major callback vector against the
analytic free-node values. This complements the packing/norm unit test by
exercising the actual ExprLegacy callbacks end to end.

The complete module Debug run including ignored workloads passed all 21 tests
(18 contract/adapter tests and 3 KiThe/combustion gates). The 26-state
combustion fixture converged in 3 iterations (weighted correction `0.3537259531`,
residual `2.943759142e-4`); the six-state cross-solver fixture converged in 3
iterations at both 64 and 128 nodes (residuals `3.297111284e-8` and
`1.649171449e-8`). The exact six-state KiThe fixed-grid fixture converged in 1
iteration (residual `1.041681117e-16`). These confirm the mapping/state-machine
changes on current workloads, but do not establish complete Cantera parity;
the broader stress matrix and Release gates remain open.

Added `scalar_transition_trace_matches_independent_cantera_contract_model`.
The test-side scalar model is separate from the nalgebra solver and follows the
pinned `MultiNewton` acceptance, `sqrt(2)` reduction, strict Jacobian-age and
four stale-Jacobian retry rules. The test compares every residual callback
state and Jacobian refresh state, the accepted state and correction, all trial
damping factors, per-iteration correction norms/ages, retry count, counters
and terminal typed status. A separate test pins iteration-limit exhaustion
after three accepted steps and preservation of the last accepted state. This
closes the narrow transition trace gate; it does not claim equivalence for
Cantera's finite-difference `MultiJac` assembly, which the symbolic adapter
intentionally replaces.

## 2026-11-05: Dense linear-workspace boundary

Moved Dense LU ownership and in-place RHS solves behind a Cantera-local linear
workspace. The Newton loop no longer stores or invokes `nalgebra::LU`
directly; one factorization remains reused for the base correction and every
damping-trial RHS. Invalid matrix shape and linear-backend failures have a
typed boundary, while singular solves preserve the existing typed error and
partial report. This is an architectural seam only: the selected backend is
still Dense and the Jacobian callback still returns a dense matrix. Sparse and
Banded support must use coordinate-native Jacobian data and must not densify
or convert a dense callback result to triplets.

Verification: `cargo test --lib numerical::BVP_Cantera::tests:: -- --quiet`
passed 18 tests with 3 workload gates ignored. The full Debug module run with
ignored gates enabled passed all 21 tests, including the 26-state combustion,
six-state 64/128-node combustion and exact KiThe six-state fixed-mesh cases.
Their numerical observations match the preceding section; no performance
claim is made for the workspace extraction. Sparse/Banded, AtomView/AOT,
release-matrix and perf-gates work remains open.

## 2026-11-05: Coordinate Jacobian callback contract

Added `CanteraJacobianPattern`, retaining validated coordinates in caller-
declared order, and a callback form that writes Jacobian values into a buffer
owned by the solver. The current Dense workspace scatters these values into a
matrix; this keeps the new interface testable without claiming native Sparse
or Banded support. Duplicate/out-of-range coordinates, pattern/state dimension
mismatch, wrong value length and non-finite values have typed errors. Callback
and pattern validation are performed outside the Newton trial loop except for
the required numeric Jacobian refresh; the values buffer is reused across
refreshes.

Verification: `cargo test --lib numerical::BVP_Cantera::tests:: -- --quiet`
passed 21 tests with 3 ignored workload gates. The coordinate path solved a
2-state diagonal system with the expected iteration/factorization/solve
counters, and a non-finite coordinate value returned a typed assembly failure
with the partial report. Native Sparse/Banded consumption and symbolic
ExprLegacy/AtomView callback plans remain unimplemented; no performance claim
is made.

## 2026-11-05: Coordinate-native Dense/Sparse/Banded backends

Extended the fixed-pattern callback route with a Cantera-local workspace for
Dense, production Sparse, and automatic Banded factorization. Sparse and
Banded fill preallocated structural entries directly; neither converts a
dense matrix into triplets nor allocates a dense global matrix. Banded widths
are inferred from the validated coordinate pattern. The solver report names
the effective backend, and structural-backend selection with a dense-only
callback is rejected as a typed configuration error.

Timing telemetry now separates coordinate assembly from factorization and is
only clocked in `Timings` mode. `Off` remains the default and reports no
counters or durations. Native linear-kernel errors retain their typed
`BvpSciNewError` source, and solve errors retain the partial Cantera report.

Verification: `cargo test --lib numerical::BVP_Cantera::tests:: --
--include-ignored --quiet --test-threads=1` passed 26 tests, including the
ignored combustion fixtures. The same three-state tridiagonal coordinate
system converges with Dense, Sparse and inferred Banded and returns matching
solutions, backend identities and counters; timing tests confirm assembly and
factorization are separately measured. This is numerical callback-path
coverage only: production ExprLegacy coordinate preparation, AtomView, AOT,
large symbolic backend parity and performance gates remain open.

## 2026-11-05: Initial BVP_api onboarding

Added Cantera as an explicit route in the existing `BVP_api`, not as a second
facade. The route validates a uniform fixed mesh and the interval-column guess
shape, delegates preparation and solve to the native Cantera adapter, and
publishes a node-major profile only after success. The native solve report is
retained in `BvpSolution::Cantera`; `BvpSolutionView` and common postprocessing
read the reconstructed full profile. Fixed-parameter binding, parameter loops
and mesh restart return typed unsupported-route errors rather than suggesting
that Cantera supports them.

Telemetry remains route-native. Before the first numerical attempt, the facade
reports `Cantera(None)` rather than a synthetic all-zero snapshot. After a
successful or numerically failed attempt it exposes the native partial snapshot.
The facade rejects `Timings`-only because Cantera's native `Timings` mode also
collects counters; `Detailed` maps to that combined mode. Logging continues to
use the shared `SolverLogLevel` and defaults to Off.

Verification: `cargo test --lib numerical::BVP_api::tests:: -- --quiet` passed
27 tests, including Cantera success, typed-failure and partial-telemetry
contracts. A bounds-blocked failed solve returns its typed native error and
partial counters without publishing a profile. `cargo run --example
bvp_api_cantera --quiet` completed the 5-node constant-profile example and
reported 2 residual calls, 1 Jacobian call, 1 factorization and 2 linear solves.
The full Cantera module Debug gate, including ignored combustion fixtures,
passed 26/26 tests at that checkpoint. This is facade onboarding evidence only;
it does not make the pilot first-tier production-ready or establish
release/performance parity.

## 2026-11-05: Direct symbolic coordinate preparation

Replaced the temporary `NRBVP::try_eq_generate` preparation bridge with a
Cantera-owned `symbolic_plan` module. It uses the prepared BVP_sci Lambdify
contract for pointwise RHS and structural `df/dy`: ExprLegacy remains the
default, and AtomView uses native packed-Atom differentiation without an
Expr-roundtrip. Cantera owns the fixed-grid trapezoid residual, reduced
endpoint mapping and global coordinate pattern. No Damped or BVP_sci nonlinear
solver object is constructed. Preparation errors retain `BvpSciNewError` as their typed
`Error::source`; the missing-callback invariant has a distinct typed contract
error.

The symbolic route now selects Dense, Sparse or automatic Banded directly from
`CanteraExprBvpOptions`. The global pattern is built once from nonzero local
derivatives and free/fixed node mapping. Sparse/Banded assembly consumes
coordinate values without a global dense Jacobian. Residual, full-grid, RHS,
argument, local-Jacobian and coordinate-value scratch buffers are reused; the
residual callback writes into solver-owned output storage. This removes
explicit per-callback scratch/result allocations from the ExprLegacy bridge,
but is not yet a measured performance claim.

Verification: scalar `y'=-y` matches the closed-form trapezoid solution across
ExprLegacy/AtomView and Dense/Sparse/BandedAuto. A coupled two-state problem
with mixed endpoint constraints matches the complete coordinate Jacobian to
finite differences for both frontends. The KiThe six-state fixed-mesh
AtomView/Sparse fixture passes its bounds and residual gates; the 26-state
combustion workload passes on AtomView/Sparse in 3 Newton iterations. The
callback buffer test confirms solver-owned residual storage keeps the same
address across evaluations. The full ignored Debug matrices passed after this
frontend addition: BVP_Cantera 32/32 (including KiThe and large combustion
fixtures), BVP_api 28/28 (including both symbolic frontends over
Dense/Sparse/BandedAuto). AOT, backend failure cross-product, release
matrices and perf-gates remain open.

## 2026-11-05: Fixed-input binding and callback reuse

Added a fixed-input parameter ABI to the prepared Cantera symbolic route.
External names are inferred from equation symbols after excluding the state
and independent variables. `bind_parameters(names, values)` validates the
schema and resolves caller ordering once; `set_parameter_values` then updates
the solver-owned values consumed by residual and Jacobian callbacks. This
keeps the structural coordinate Jacobian, Lambdify closures and generated AOT
artifact prepared across re-solves. Missing values and invalid schemas remain
typed configuration errors. The shared BVP facade and Python request now
expose the same binding/rebinding operation; Cantera parameter loops remain
unsupported and distinct from ordinary rebind.

Verification: Debug regression for `y'=-rate*y` checks the unbound error,
initial binding, and changed endpoint after rebinding. The BVP_api regression
checks the public fixed-input lifecycle. The ignored AOT parity matrix passed
for ExprLegacy/AtomView and Dense/Sparse/BandedAuto, including a second solve
at a new parameter value. Focused Python Cantera contracts passed (`2 passed,
367 deselected`), and the Python plus Rust facade examples ran successfully.
These are correctness/API results, not a performance claim. The refreshed
six-stage Release matrix passed at
`test_reports/BVP_Cantera_release_manual/20261105T055624Z/`. Large-workload
performance gate expansion, baseline recording, and clean-source comparison
remain open.

## 2026-11-05: Jacobian callback error-source audit

Added a typed failure regression for a symbolic Jacobian callback on Dense,
Sparse and BandedAuto. In each case, the original `BvpSciNewError` remains
available as the error source, the partial report retains the state, and the
Jacobian attempt counter is one. Residual evaluations correctly remain zero:
the Newton lifecycle evaluates/factors the Jacobian before its first residual
call, so the injected failure stops before residual work. The focused Debug
test passes. This supplements the existing residual source-chain, structural
assembly, singular/linear-backend, bounds-blocked, post-acceptance failure and
nonconvergence tests; it does not claim allocator-level memory measurements.

## 2026-11-05: Large Bratu and KiThe performance fixtures

Added two dedicated large workloads to the `bvp-cantera-matrix` perf-gate
group. Bratu is represented on its symmetric half-domain (`y(0)=0`,
`y'(0.5)=0`) so its second-order boundary conditions map to the Cantera
first-order contract of one endpoint constraint per state. The fixture has
1024 intervals and now runs 20 fresh Sparse/ExprLegacy solves per sample. A
Release smoke measured 10.233 ms for the initial three-solve batch, so the
repetition count was raised to avoid a short/noisy gate. The 26-state KiThe
combustion fixture is reused from the maintained BVP regression source behind
the `perf-gates` feature, with six fresh AtomView/Sparse solves per sample and
no duplicated kinetic equations; its initial three-solve batch measured
50.849 ms. After increasing repetitions, a Release smoke measured `62.133 ms`
for Bratu and `101.279 ms` for the KiThe batch. Both correctness smoke tests pass in
Debug and Release perf-gates test profiles, including native convergence,
finite residual/profile, configured bounds where applicable, and endpoints.
These are single smoke samples, not statistical timing claims; repeated
perf-gate recording and baseline stability remain unmeasured.

The linear-workspace storage audit now also checks an 18,000-unknown
tridiagonal pattern (`nnz=53,998`). Cantera-owned Sparse and BandedAuto
coordinate/entry/scratch buffers each report `2,447,920` capacity bytes and
remain O(nnz+n); neither allocates the dense global matrix. This deliberately
excludes allocations private to the reused native Faer backend, so it is not
reported as process peak memory. Native allocator/peak-RSS accounting remains
open.

## 2026-11-05: Continuous-analytic fixed-grid correctness gates

Added two inexpensive analytic gates over ExprLegacy/AtomView and
Dense/Sparse/BandedAuto at 16, 32 and 64 intervals. The first solves
`y'=z, z'=2*y^3` against `y=1/(1+x), z=-1/(1+x)^2`, starting from a perturbed
guess. The second derives a Cantera-compatible mixed-endpoint variant from
the nonlinear `TwoPointBVP` in `Examples_and_utils.rs`: it retains the same
equations and exact `y=exp(-x^2/4), z=-x*y/2`, but uses `y(-1)` and `z(1)` rather
than the original two conditions on `y`. It must not be cited as a test of the
original boundary-condition pair.

Both gates check prescribed endpoints, cross-route profile parity, discrete
trapezoid residual and approximately second-order reduction in continuous
profile error under each mesh doubling. The first fixture's maximum profile
errors were `7.675024e-4`, `1.918373e-4` and `4.795698e-5`; its maximum
trapezoid residual stayed below `1e-16`. The derived Gaussian fixture's errors
were `1.860303e-3`, `4.650464e-4` and `1.162600e-4`; residuals stayed below
`6.4e-12`. All six frontend/layout routes produced matching profiles and
operation counts at each resolution. The tests write Tabled diagnostic reports
through the established reporting helper under `test_reports/BVP_Cantera`.

Debug verification: the complete non-ignored Cantera test module passed
`37 passed, 0 failed, 6 ignored`. These are numerical correctness results, not
performance measurements. The original `Examples_and_utils.rs` fixtures still
need an endpoint-contract audit before claiming broader analytic coverage.

## 2026-11-05: Test organization and generated compiler presets

Split the former 2,074-line test source into `tests/symbolic.rs`,
`tests/newton.rs`, `tests/telemetry.rs` and `tests/large_workloads.rs`, with a
small shared module root. The solver-level test filter remains
`numerical::BVP_Cantera::tests::`; the release script's direct ignored-AOT
selector now includes the symbolic submodule.

The shared IVP generated-backend configuration gained an optional
`Production/FastBuild/DevFastest` build-preset override. No override preserves
the existing Debug-to-DevFastest and Release-to-Production behavior. Zig's
FastBuild mapping now uses `ReleaseSmall`, matching the BVP generated-backend
preset taxonomy; C/GCC, C/TCC and Rust receive their respective existing
compile configurations. A separate ignored Cantera matrix covers both
frontends, all four compiler choices and all three presets with per-route
rebuild directories, Lambdify parity, parameter rebinding and a Tabled report.
The release matrix has a dedicated stage for it.

Release verification completed all 24 Sparse routes. GCC `16.1.0`, TCC
`0.9.27`, Rust `1.98.1` and Zig `0.17.0-dev.340+c02d39a21` each built under
Production, FastBuild and DevFastest for ExprLegacy and AtomView. All
pre-rebind and rebound state differences from Lambdify were exactly zero; each
route converged in one iteration with one Jacobian evaluation. Full AOT
compiler/profile table: `test_reports/BVP_Cantera_AOT/release/aot_compiler_and_profile_matrix.md`.
This establishes compiler/profile correctness on Sparse, not relative runtime,
compile-time or binary-size performance. The prior C/TCC AOT matrix continues
to cover Dense/Sparse/BandedAuto; its relocated Release selector was rerun and
passed (`1 passed, 0 failed`).

Preset mapping used by the current shared builder: C presets use the existing
`-O3`, `-O1` and `-O0` profiles respectively (fast/developer flags are applied
after the release baseline); Rust uses Cargo Release defaults, then
`opt-level=1/codegen-units=16`, then `opt-level=0/codegen-units=16`; Zig maps
to `ReleaseFast`, `ReleaseSmall` and `Debug`. The Tabled artifact currently
records preset names and compiler versions; printing the fully resolved flags
in that table and a Cantera-facing missing-compiler typed-error test remain
open.

Verification after the module split and preset API change: the Cantera Debug
module passed `37 passed, 0 failed, 7 ignored`; the shared generated-plan
resolver test passed and confirms explicit preset selection without changing
legacy defaults. The Release compiler/profile matrix passed. AOT cache-reuse
lifecycle and typed missing-toolchain behavior remain separate evidence gaps.

## 2026-11-05: Replace batched short perf-gate workloads

The earlier Cantera perf-gates aggregated repeated solves whose individual
durations were too short for the project's workload policy. Replaced them with
one fresh solve per sample and dimensioned the scenarios independently:
six-state combustion at 24000 intervals, symmetric half-domain Bratu at 32768
intervals, and the KiThe-derived 26-state combustion fixture at 256 intervals.
Stable scenario IDs include the dimensions and the scenario versions were
advanced, preventing accidental comparison against the former batch baselines.

The off-telemetry wall metric now stops immediately after `solve()` and covers
symbolic preparation plus one solve; profile expansion and correctness scans
are outside the timer. Detailed diagnostics remain a separate Timings pass.
The standalone bench similarly excludes AOT warmup/artifact build and stops
its solve and total timers before profile expansion/validation. Dense remains a
128-interval correctness/control row; only Sparse and BandedAuto use the large
24000-interval workload for performance interpretation.

Release calibration evidence is recorded in
`test_reports/BVP_Cantera_release_manual/20261105T113909Z/reports/BVP_Cantera_release/release/bvp_cantera_workloads.md`.
At 12000 intervals, single tabled samples measured Sparse solve at
15.873-16.230 ms and BandedAuto at 8.453-8.753 ms, with prepare+solve totals
of 36.577-45.602 ms. These values are not repeated-sample medians; the solve
alone remained below the desired several-tens-of-milliseconds target, so the
main workload was raised to 24000 intervals. Bratu at 8192 was 21.424 ms,
motivating the increase to 32768. The KiThe 26-state/256-interval route passed
at 134.313 ms for one fresh prepare+solve. These 12000-size measurements are
calibration history only, not baselines or regression verdicts; final-size
results follow.

Final-size follow-up completed the Release registry checks and the Tabled
benchmark on 24000 intervals. All 12 rows passed. Across ExprLegacy/AtomView
and Lambdify/AOT, Sparse solve was 33.247-34.000 ms, BandedAuto solve was
18.544-19.291 ms, and symbolic preparation was 58.703-60.587 ms. The Dense
128-interval control took 12.655-13.198 ms to solve and is not a performance
case. Maximum reported analytic error was `1.367e-12` for the large fixture;
all routes used one Newton iteration with matching callback/factorization
counts. The separate Release registry smoke passed 5/5: 24k combustion routes,
32k Bratu (89.279 ms), KiThe 26-state/256-interval combustion (124.587 ms),
and the ten-point parameter continuation. The table is at
`test_reports/BVP_Cantera_release_manual/20261105T122507Z/reports/BVP_Cantera_release/release/bvp_cantera_workloads.md`.
These are single tabled measurements and smoke timings, not repeated medians or
baseline/regression verdicts; stable perf-gate samples on a clean revision
remain necessary.

## 2026-11-05: Parameter continuation, task clients, and examples

Added native `continue_with_parameter_values` and ordered
`solve_parameter_loop` for prepared symbolic Cantera BVPs. Continuation starts
from the last successful checkpoint (or the configured initial guess before
the first success), while a loop always starts its first point from that
configured guess and then carries only successful profiles forward. Input
vectors are fully preflighted before callback work; failed points keep typed
errors and partial native telemetry, later points still run, and the solver
restores the last successful loop checkpoint. If every point fails, the exact
entry state/parameters and any previously accepted profile are restored. The
prepared symbolic Jacobian, callbacks and AOT lifecycle are reused.

The same route is available through `BVP_api` and Python's prepared BVP facade.
Python returns detached node-major profiles and explicit per-point status,
error, guess provenance and telemetry maps; absent telemetry remains distinct
from measured zero values. The full BVP task-document parser now constructs and
runs the Cantera route, and the settings-only hybrid parser can construct its
native options without client equations. Unsupported Cantera task features
remain typed errors rather than being silently discarded.

Debug verification: all 57 active `BVP_Cantera` tests passed (7 ignored); all
three facade continuation tests passed, including a six-state 2048-interval
matrix across both frontends and Sparse/BandedAuto; all 5 Cantera full-task parser
tests and the hybrid settings-only test passed. Both Rust examples completed
ten-point sweeps on 257-node meshes. The combustion-like six-state fixture had
discrete trapezoid residuals no larger than `1.16e-11`; the Bratu sweep matched
the analytic lower branch with maximum profile error below `8e-6`. The local
Windows editable Python extension built successfully, and its focused suite
passed 7 tests across both frontends and Off/Counters/Detailed telemetry; both
Python examples also completed all ten points. These are correctness checks,
not performance baselines. The Cantera release matrix now explicitly runs the
full-task/hybrid parser tests and both continuation examples. Clean-source
repeated perf-gate record/compare remains open.

The registered `bvp-cantera-parameter-continuation` gate was updated in this
cycle to call the native `solve_parameter_loop` rather than manually repeating
`set_parameter_values` plus `solve`. It checks all ten point records, strict
ordering, first-guess/last-success provenance, every full profile and each
analytic discrete endpoint. The scenario version advanced to `2`, so the new
lifecycle measurement cannot be compared accidentally with the earlier
manual-loop baseline. Its registry and one-run correctness smoke pass; a new
repeated baseline/compare is still required on a clean source revision.

The updated twelve-stage Release matrix passed on Windows at
`test_reports/BVP_Cantera_release_manual/20261105T170539Z/`. It reports 64/64
Cantera tests including ignored AOT cases, 32/32 BVP_api tests, 5/5 perf-gate
correctness/registry tests, 5/5 full task-parser tests, and 1/1 hybrid-parser
test. Python binding compilation, the facade and both ten-point continuation
examples, the 12-route Tabled benchmark, AOT frontend/layout parity, and the
AOT compiler/profile matrix all passed. The compact stage/duration table is in
`reports/release_summary.md`; detailed outputs are under `technical/`.
These checks close the current correctness/reporting gates, but do not create a
performance baseline. Initial repeated record/compare still requires a clean
source revision and stable samples.
