# Performance Regression Gates

Status: opt-in runner and initial Windows baselines in progress, 2026-10-21.
It currently contains 199 scenarios, including full supported-route matrices for Radau (24 cells),
BDF (8 cells), LSODE2 (30 cells), BVP_Damp (66 cells), BVP_sci (30 cells),
BE (4 cells), Nonlinear_systems (6 cells), optimisation (14 cells), and a
separate ten-cell `bvp-large` group. Initial BDF, Radau and BE baselines have
been recorded and compared on the Windows development machine; large BVP
baselines are pending. The runner is not yet a complete cross-platform
release-quality regression gate.
See [TODO.md](TODO.md) for validation and coverage gaps.

This repository tool will detect performance changes in selected, fixed
RustedSciThe workloads on the same registered machine. It complements numerical
correctness and algorithmic parity checks. It does not replace their verdicts.

## Ownership

| Artifact | Purpose | Update policy |
| --- | --- | --- |
| Existing solver STORY_TESTS Markdown | Research history, interpretation, engineering decisions | Developer edits |
| Existing `test_reports/` | Experimental matrices and stage diagnostics | Existing tests and scripts |
| `performance/machines/` | Reviewed machine profiles | Explicit registration |
| `performance/baselines/` | Machine/environment/scenario performance references | Explicit `record`, overwrite requires `--force` |
| `performance/runs/` | Individual gate reports and raw measurements | Runner output, Git-ignored |
| `performance/local/` | Optional local working artifacts | Git-ignored |

The local machine UUID belongs in the OS user configuration directory, outside
the checkout. Do not put usernames, absolute personal paths or hardware serial
numbers into versioned profiles.

## Runner and Isolation

Use one explicitly declared, opt-in Cargo binary named `perf-gates` in the
existing package, with its source under `performance/runner/`. No new workspace
crate is needed. Keep CLI, registry/workloads, environment, measurement,
comparison/storage and reporting in a few cohesive private modules.

Neither the library module tree nor numerical APIs depend on these modules.
Existing benchmarks and STORY tests keep their independent entry points.
Reuse public workload definitions and solver APIs; extract small fixture
builders only when needed and after checking existing test semantics.

The required profiling axes are kept orthogonal: symbolic frontend
(`ExprLegacy`/`AtomView`), execution (`Lambdify`/`AOT` where implemented),
linear algebra layout (`Dense`/`Sparse`/`Banded` where implemented),
dispatch (`Sequential`/`Parallel` where implemented), and timing scope. A
scenario is a supported route, not a synthetic Cartesian-product cell: the
registry must not advertise combinations the solver does not implement.
Matched frontend or policy comparisons use identical equations, dimensions,
layout, tolerances, and lifecycle except for the named axis.

The current registry reuses the existing BDF benchmark helper for BDF, the
shared IVP equation corpus and public APIs for LSODE2/Radau, BE benchmark
support, a nonlinear workload from the Criterion matrix, BVP_sci full-solve
routes, and one faithful compact banded solve. Its current matched
ExprLegacy/AtomView pairs cover BDF, Radau, LSODE2, BVP_Damp, BVP_sci, and BE. BDF additionally has a fixed
64-state diffusion matrix covering both frontends with Lambdify
Sequential/Parallel and AOT Whole/Parallel2. BDF uses its fixed Dense
implementation and does not expose a selectable layout axis. AOT uses a
runner-owned BuildIfMissing cache, `tcc`, and warmups. Radau additionally has a
complete fixed-workload matrix for the supported combinations:
ExprLegacy/AtomView x Lambdify/AOT x Dense/Sparse/Banded x
Sequential/Parallel. Its AOT route uses a runner-owned BuildIfMissing cache,
requires `tcc` on PATH, and uses at least one warmup to keep cold compilation
out of measured warm samples. LSODE2 has a fixed 32-state diffusion matrix for
ExprLegacy/AtomView x Dense/Sparse/Banded x Lambdify Sequential/Parallel/Auto
and AOT Whole/Parallel2. It fixes LSODE2 to its BDF-only faithful controller
so the matrix isolates frontend, execution, layout, and callback policy rather
than mixing controller changes. The original small Robertson scenario remains
as a separate low-cost representative gate.

For BVP_Damp, 44 Damped cells cover the existing oscillator and nonlinear
dashboard workloads across both frontends, all three layouts and Lambdify
Sequential/Parallel/Auto plus supported Sparse/Banded AOT Auto. Frozen adds 22
cells on the exact linear fixture already used by its Rust correctness tests;
it has a distinct boundary-condition contract from Damped. AOT continuation
re-prepares after each numeric parameter rebind because the current prepared
AOT runtime invalidates on rebind; `continuation_prepare_ms` makes that cost
explicit. Dense AOT is intentionally absent. `list` reports workload,
frontend, execution route, linear algebra layout, dispatch policy and telemetry
contract for every registered scenario. The representative nonlinear matrix
is described below; broader method-specific matrices and remaining optimization
routes remain explicit gaps in TODO, not implied coverage.

BVP_sci adds 30 scalar exponential full-solve cells: ExprLegacy/AtomViewNative
x Dense/Sparse/Banded, with Lambdify Sequential/Parallel/Auto and AOT
Sequential/Parallel. Every route checks exact-solution error, success status,
Off-versus-Detailed parity, and actual AOT selection where requested. The
scalar Banded route uses `(0,0)` bandwidth; block bandwidth belongs to larger
systems. The optional SuperLU bridge is deliberately excluded from this
portable matrix so Sparse measures StandardFaer.

The `bvp-large` group adds six-state full solves to avoid calibrating BVP
baselines from tiny scalar tasks. BVP_sci uses its established combustion-like
story equations on a fixed 257-node mesh with Sparse/Faer across
ExprLegacy/AtomView and Lambdify/AOT. BVP_Damp uses the separate chemical
combustion-1000 fixture with a curated Sparse/Banded frontend and execution
set. These are not a cross-solver parity pair. Each timed sample is
telemetry-Off, then a separate Detailed run checks counters and solution
parity. These large workloads have their own IDs and baselines, leaving the
compact route matrices intact.

BE adds four matched parameterized-combustion routes: ExprLegacy/AtomView x
Lambdify/AOT. Dense linear algebra and sequential execution are fixed properties
of this API, not missing matrix cells. Each route checks conservation,
nonnegative finite states, final time, Off-versus-Detailed trajectory parity,
and that the requested AOT runtime is actually selected. Separate cold AOT
preparation/build/link and warm callback timings are emitted only when native
telemetry reports those stages.

Nonlinear_systems adds six matched dense Newton routes on the same 64-variable
quadratic-chain system with a known root: both frontends use Lambdify
Sequential/Parallel, while AOT is represented once per frontend because its
generated route has no callback-dispatch option. The fixed Dense implementation
is not expanded into fictitious Sparse/Banded rows. Every route validates the
root, residual and Off-versus-Detailed parity; preparation and native solve
stages are reported from the Detailed diagnostic run.

Optimisation adds 14 symbolic fitting routes on one deterministic
201-observation exponential/sinusoidal dataset. Gavin covers ExprLegacy and
AtomView for its three damping-update policies; classic `FittingSolver` covers
both frontends with Lambdify Sequential/Parallel; VarPro is represented through
its native ExprLegacy/Sequential API; and UniversalFitting gets a separate
route for each of LM, Gavin, and VarPro. No AOT or unsupported VarPro frontend /
dispatch cells are fabricated. Each route checks recovered coefficients,
R-squared, and Off-versus-Detailed result parity. Detailed callback and linear
stages are included only where the native telemetry type reports them.

The primary wall-clock gate must use telemetry Off. Every telemetry-Off
warmup and gate sample is collected before instrumented diagnostics begin.
Metric records preserve whether a value came from the gate or a diagnostic
pass, plus a runner-local timing-scope label and optional parent scope. A
reported zero remains zero; a structurally absent stage remains absent.
Radau stage timings are
collected by a separate `Timings` solve; BDF and LSODE2 use `Detailed` to retain
both stage timings and discrete work counters. These instrumented diagnostics do
not affect the Off-mode wall-clock gate. Other solvers should follow this
pattern using `Timings` or `Detailed`, according to native support. Reports must retain both views
without claiming that instrumented wall time is the uninstrumented runtime.
Each registry entry explicitly names its regression-gate timing. Initially,
`scenario_wall_ms` is the gate; preparation, setup, factorization and solve-stage
timings are retained as diagnostics and do not independently trigger a
regression verdict. Gate IDs are part of the versioned baseline contract.
Further solver and symbolic scenarios are
incremental; the runner does not enumerate a full experimental cross-product.
Run artifacts are written only below the Git-ignored `performance/runs/` tree.
Existing `test_reports/` and solver STORY_TESTS remain independent and are not
read or modified by the runner.

## Commands

Use Release for every CLI command for a consistent machine fingerprint. The
`list` and `machine register` commands do not run numerical workloads, but
using the same profile avoids accidentally registering a Debug environment.

```text
cargo run --release --features perf-gates --bin perf-gates -- list
cargo run --release --features perf-gates --bin perf-gates -- machine register --name home
cargo run --release --features perf-gates --bin perf-gates -- list --solver bdf
cargo run --release --features perf-gates --bin perf-gates -- record --scenario bdf_robertson --machine home
cargo run --release --features perf-gates --bin perf-gates -- compare --scenario bdf_robertson --machine home --dry-run
cargo run --release --features perf-gates --bin perf-gates -- compare --solver bdf --machine home
cargo run --release --features perf-gates --bin perf-gates -- compare --all --machine home
```

`record` and `compare` require an explicit selector. An unknown selector is an
error. `compare` never writes a baseline; missing, incompatible or unstable
measurements cannot become PASS.
Baseline recording and live comparison reject Debug binaries. Debug builds
remain appropriate for runner unit tests and correctness-only smoke checks.
`record` also refuses a dirty or unversioned source tree by default. Generated
files under `performance/baselines/` are excluded from this source-dirty check:
they are measurement outputs, so a multi-solver baseline session does not
require a commit after each solver. Changes to runner code, scenario
definitions, machine profiles, or other source inputs still require a clean
tree. Commit reviewed baselines when they are ready to share.
`--allow-dirty-source` is an explicit diagnostic escape hatch; the
baseline records the revision, dirty flag and lockfile fingerprint, so that
such a reference is visibly non-clean rather than silently misrepresented.

Machine registration creates a local installation identity under the user's
OS configuration directory and a small alias profile under
`performance/machines/`. Review the profile before committing it. Runtime
provenance records source revision, dirty-tree state, Cargo.lock content
fingerprint and `rustc --version` from `PATH`. This does not prove which
compiler built an already-existing executable. Compiler version and source
revision are reported but are not hard compatibility keys: changed code and
toolchains are exactly what later comparisons should measure. They must
nevertheless be recorded in reports so results can be interpreted.
Build-time compiler and artifact provenance remain explicit limitations, not
inferred facts.

An uninstrumented solve and an instrumented diagnostic solve have separate
measurement contracts. Preparation, callbacks, numerical solve and continuation
must retain their declared scopes. Baseline schema 3 persists metric phase and
scope metadata; schema changes are rejected rather than reinterpreted.
Baselines describe the actual frontend,
execution route, linear backend, telemetry mode and cache lifecycle.

[`METRIC_CONTRACTS.md`](METRIC_CONTRACTS.md) maps emitted diagnostic IDs to
native fields and records units, route applicability, known aggregation
boundaries and overlaps. Native parent/child scope metadata is persisted where
the solver exposes it; unknown reset semantics are called out rather than
inferred.

Metrics are declared by each scenario; the runner does not add a wrapper timer
that would accidentally include correctness checks. Metric names and units
must remain identical across samples. Timings retain raw samples plus
median/MAD/min/max; discrete counters retain raw samples plus median and range.
Counter changes are reported as algorithmic-work diagnostics, not silently
converted into timing failures. Correctness diagnostics are stored separately.

Setup, solve and full-solve scopes may overlap by design. For example,
`scenario_wall_ms` must not be added to nested setup and solve values. Failed
scenarios and partial matrix outcomes are summarized in `summary.md` and
`summary.toml` under the run directory, without touching `test_reports/` or
STORY_TESTS.

Before the symbolic refactor, complete the minimal runner and record reviewed
release baselines from an identified source revision. Debug builds suffice for
runner logic and numerical validation; performance evidence requires release.
