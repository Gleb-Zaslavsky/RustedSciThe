# RustedSciThe Architectural Guideline

This document defines the repository-level expectations for numerical solvers,
their telemetry, story tests and performance evidence. It is intentionally
practical: a solver is not production-ready merely because its numerical core
returns plausible values. Its lifecycle, failure modes, observability and
repeatable evidence must also be understandable to a user and maintainer.

The Radau release infrastructure is the current reference implementation for
these rules. Existing solvers may migrate incrementally, but new solver work
should follow this guideline from the beginning.

## Testing And Benchmarking

### 1. Telemetry Is Part Of The Solver Contract

Every solver should provide a developed telemetry model covering both discrete
and timed observations.

Discrete telemetry should include, where applicable:

- residual, Jacobian, Newton and linear-solve call counts;
- matrix factorizations, refactorizations and linear solves;
- accepted, rejected and failed steps or iterations;
- symbolic preparation, differentiation, simplification and pattern work;
- frontend/backend builds, links, cache hits, cache misses and publications;
- parameter rebinds, continuation calls, worker dispatches and chunks;
- copies, workspace growth, materialization events and output writes.

Timed telemetry should identify the important stages separately, for example
symbolic preparation, callback evaluation, Newton/controller work, matrix
assembly, factorization, real/complex solves, compilation, linking and
publication. A parent scope and its child scopes are diagnostic scopes and
must not be added together unless the report explicitly says that they are
disjoint.

The normative collection levels are `Off`, `Counters`, `Timings` and
`Detailed`. Their meanings are fixed across solver APIs:

- `Off` performs no counter updates or clock reads;
- `Counters` records discrete events without starting timers;
- `Timings` records durations without updating discrete counters;
- `Detailed` records both counters and timings.

Preparation telemetry is a separate lifecycle scope when it has a separate
owner; its enablement and cost must be named explicitly rather than silently
folded into numerical-solve telemetry. A solver that cannot distinguish one
of the four levels must reject or explicitly document that unsupported level,
not silently alias it to a different collection policy.

The shared Rust API type is `numerical::SolverTelemetryMode`. The nonlinear
systems engine and Gavin LM use this contract and default to `Off`. Existing
solver-specific telemetry enums remain supported during migration, but their
variant semantics must be documented and audited against this contract rather
than assumed equivalent from similar names.

Migration note (2026-10-16): the nonlinear systems engine, Gavin LM, and
ShootingBVP implement all four levels. ShootingBVP exposes the selected mode
in its telemetry snapshot, leaves unrequested channels untouched, and its
builder and symbolic/native solve routes use the shared enum. Python currently
exposes `off`/`counters`/`timings` for this solver; adding Python's `detailed`
spelling belongs to the broader binding-contract migration.

Telemetry must be:

- off by default, with each additional collection level selected explicitly;
- non-invasive on the hot path when disabled;
- safe to enable for correctness and performance diagnostics;
- typed and stable enough for story tests and report consumers;
- explicit about unavailable, inapplicable and zero-valued measurements.

The values `0`, `-`, `unknown` and `not applicable` must not be used
interchangeably. For example, zero parallel dispatches means something
different from a policy for which worker telemetry is not available.

Process-wide allocator measurements must not be represented by a logical
workspace or materialization counter. If allocator statistics are not
available, the report must say so.

### 2. Logging Is An Explicit, Persistent Output Channel

Logging uses the `log` facade with a concrete backend such as `simplelog`.
It is distinct from telemetry: telemetry is a typed in-memory snapshot of
discrete counters and/or durations, while logging is a human-readable event
stream. Enabling one must not silently enable the other.

Logging must be disabled by default and explicitly configurable by the caller.
`Off` emits no solver log. Any enabled severity (`Debug`, `Info`, `Warn` or
`Error`) must have a persistent text-log destination; writing only to an
in-memory event buffer or relying on console output is not sufficient. A
timestamped file per process is the default lifecycle: enabled solvers share
that process log rather than each creating a competing global logger. Its
location must be discoverable by the caller. If logging is configured but
the file cannot be created, report that setup failure rather than silently
discarding the requested log.

For complex solvers, `Info` or a more verbose mode may include a concise
`tabled` or `prettytable-rs` summary at selected iterations or at solve
completion. Such summaries should expose the useful subset of timings and
discrete counters, not dump every internal field. Iteration-level detail is
appropriate for diagnosing convergence; an end-of-solve summary is preferred
for routine runs. Warnings and errors should preserve actionable context and
must also reach the configured text log.

### 3. Ported Algorithms Require An Auditable Parity Reference

When a solver ports a mathematical algorithm from another language, keep a
small, attributed subset of the upstream implementation in a local
`*_reference` directory. Record the upstream project and exact version or
revision, and include only the modules needed to establish the algorithmic
reference. Do not make the reference copy a runtime dependency of the Rust
solver.

Instrument the reference route and the Rust route with clearly defined,
semantically matched telemetry. Parity checks should prioritize discrete
algorithmic counters such as residual and Jacobian evaluations, accepted or
rejected steps, Newton iterations, LU factorizations and linear solves.
Counter definitions must be documented before comparing values: equal names
do not imply equal semantics. Stage timings are useful to explain performance
differences, but are diagnostic evidence rather than a requirement for exact
timing equality. Compare equivalent lifecycle scopes and workloads.

The port should preserve the source algorithm's mathematical controller and
numerical decisions unless a deviation is deliberate, justified and recorded
in a stage-by-stage parity ledger. Unexplained counter or behavior differences
are gaps to investigate, not presumed optimizations.

The reference boundary is the mathematical algorithm, not every implementation
detail of the source language. Rust routes should use the crate's native
symbolic and numerical infrastructure: Lambdify/AOT for residuals and
Jacobians where supported, and established Rust linear algebra backends such
as nalgebra Dense, Faer Sparse and the crate's Banded solvers. Do not copy a
source-language linear algebra backend merely to claim parity. Backend
differences are acceptable when correctness is established and their effect
on counters, stability and performance is measured and explained.

### 4. Every Solver Has A Broad Evidence Corpus

Each solver should have both ordinary story tests and Criterion benchmarks.
The corpus should cover the solver's real axes rather than only one scalar
smoke problem:

- small nonlinear and stiff problems for fast debug correctness checks;
- representative medium and large workloads;
- independent numerical references and parity between supported frontends;
- every supported matrix layout and backend;
- analytic and numerical Jacobian paths where supported;
- fresh solve, warm solve, restart and parameter continuation;
- failure, timeout, invalid-shape, non-finite and exhaustion paths;
- sequential, parallel and automatic policies where supported;
- lifecycle/cache/provenance and process-isolated producer/consumer paths;
- preparation-only, callback-only, full-solve and stage-attribution views.

Story tests are the executable explanation of the solver's behavior. They
should state what is being compared, what is intentionally not comparable and
which counters establish that the two routes followed the same lifecycle.
Performance stories should report absolute times and relevant counts, but
should avoid brittle portable wall-clock assertions unless the environment is
controlled and the threshold is justified.

For execution policies, a solver that supports parallel work must have a
matched policy matrix, not only a successful parallel smoke test. At minimum
the matrix should compare Sequential against Parallel and Auto on the same
workload, and compare chunked against whole or unchunked execution for
generated backends where chunking exists. The matrix should include both
callback-only and full-solve views. It must report configured and observed
worker counts, dispatch counts, chunk counts, worker callbacks and whether
parallel execution was applicable. Auto results may remain sequential; that is
a valid fallback and must be visible rather than reported as a failure.

A portable break-even claim requires repeated measurements over more than one
workload size or an explicit statement that no portable crossover was found.
A policy matrix is a correctness/performance diagnostic, not permission to
assert that Parallel is universally faster. Thread startup, scheduling and
chunk materialization can dominate small systems.

### 5. Compact Tables Are The Primary Human-Readable Output

Story tests and release-oriented benchmarks must collect their meaningful
values and print a compact `tabled` summary. The table should contain stable
column names, route keys, workload/layout identifiers, status and the numbers
needed to interpret the result. Do not make a human search through callback
warm-up output, compiler progress or repeated Criterion chatter.

For policy/chunking matrices, stable columns should additionally include
policy, chunking, configured_workers, observed_workers, parallel_applicable,
parallel_dispatches, sequential_dispatches, chunks, worker_callbacks,
callback_ms and full_solve_ms where those measurements are available. Use a
typed applicability field or an explicit dash when a metric does not apply; do
not encode unavailable data as zero.

Reports with numerical results should be written through
`crate::Utils::test_reporting` or the repository's equivalent reporting
helper. A report should include enough metadata to reproduce the measurement:

- debug or release profile;
- host/toolchain/compiler when relevant;
- workload, dimension, matrix layout and frontend/backend;
- execution policy and configured/effective worker counts;
- tolerance/output policy and continuation count;
- lifecycle mode and cache/provenance identity where applicable.

Correctness tables should retain the numerical evidence needed to audit the
claim: maximum difference, drift, invariant residual, endpoint values or
checksums. A status-only `ok` is not sufficient for a numerical parity gate.

### 6. Reports And Technical Logs Are Separate Artifacts

The useful table and the technical process transcript have different users and
must be stored separately.

The report directory should contain:

- compact Markdown or text tables intended for review;
- one report per logical test, benchmark group or matrix;
- profile-qualified and timestamped paths so debug runs cannot overwrite
  release evidence;
- completion status, including partial or interrupted campaigns.

A separate technical-log directory may contain Cargo warnings, compiler
output, Criterion warm-up messages, progress chatter and stack traces. That
output is diagnostic only. It must not be concatenated into the compact report
unless table generation failed and the failure itself needs investigation.

An interrupted or empty campaign is not a passing gate. The summary must
distinguish `passed`, `failed`, `planned`, `interrupted` and `not applicable`.

### 7. Long Campaigns Must Be Non-Fail-Fast

Every solver with a substantial release corpus should have one or more scripts
that run the important tests and benchmarks sequentially. The script may
accept environment variables or parameters for dimensions, workloads,
policies, worker counts, continuation lengths and toolchains.

The orchestration contract is:

1. Each step has its own technical log and, where applicable, its own compact
   result report.
2. A failed test, benchmark, compiler route or toolchain does not stop later
   steps.
3. The final summary lists every step, exit code, duration, report path and
   failure reason.
4. A caller can request a non-zero final exit code after the whole sequence
   with an explicit fail-on-any option.
5. A plan-only mode can show the queue without running Cargo.
6. Release and debug outputs are profile-qualified and never silently replace
   one another.
7. A policy/chunking matrix can be selected independently from the broader
   workload matrix, so a cheap correctness smoke does not require the full
   overnight performance campaign.

This is especially important for overnight runs. A single failure should be
investigated, but it should not discard evidence from the other workloads.

### 8. Compact Release Evidence And Statistical Baselines Are Different

Compact release matrices answer questions about coverage, absolute wall-clock
cost, lifecycle correctness and numerical outcomes. They are usually one
bounded measurement per row and are therefore easy to review, archive and
compare by eye.

Statistical performance baselines require repeated measurements, normally via
Criterion or an equivalent harness. They must keep the same:

- release profile and compiler/toolchain;
- machine and relevant process environment;
- workload, dimension, layout and frontend/backend;
- worker count and execution policy;
- continuation count and output/tolerance policy.

Regression decisions should use a robust estimator such as the median and the
reported confidence interval, together with an absolute materiality rule.
Percentage-only gates are inappropriate for tiny sub-millisecond operations.
Conversely, a modest percentage change in a repeatedly executed multi-second
stage may be important. Establish at least one clean baseline archive before
introducing hard thresholds.

The current Radau statistical runner is
`scripts/radau_statistical_baseline.ps1`; its Criterion sample size and
measurement window are controlled by `RADAU_BENCH_SAMPLE_SIZE` and
`RADAU_BENCH_MEASUREMENT_SECONDS`. Its compact release counterpart is
`scripts/radau_release_matrix.ps1`.

### 9. Three Complementary Measurement Records

Optimization and solver-stabilization decisions use three complementary
sources of evidence. They answer different questions and must not replace one
another.

1. **Solver matrix scripts and `test_reports/`.** Solver-specific scripts run
   selected release story tests and benchmarks as reproducible matrices. They
   write review-friendly, compact `tabled` reports under timestamped
   `test_reports/` directories, keeping compiler/process transcripts separate
   from numerical results. These reports capture the detailed measurements of
   a particular run: workloads, routes, stage timings, counters, parity and
   status. They are run artifacts, not a substitute for the solver's curated
   engineering history.
2. **Solver `STORY_TESTS*.md` documents.** These are the durable, human-curated
   research and decision record. They explain why a change was made, which
   correctness tests support it, and what measured results justify or reject
   the hypothesis. Record meaningful numerical results, not only pass/fail:
   comparisons of architectures, symbolic frontends, linear-algebra backends,
   stage timings, counters, continuation behavior and full-solve wall time all
   belong here when they informed a decision. Link to the detailed
   `test_reports/` artifact where appropriate. Ordinary script runs must not
   automatically rewrite STORY documents.
3. **`perf-gates`.** This opt-in runner measures a curated set of stable,
   addressable workloads against a machine/toolchain-specific baseline. Its
   primary regression verdict uses telemetry-Off wall time; correctness,
   counters and instrumented stage timings provide diagnostic context. It is
   for repeatable regression control, not exploratory solver research, not a
   replacement for story/benchmark matrices, and not a cross-machine
   comparison service. Machine profiles and baselines are local; compare only
   compatible workload and environment identities. `record --force` replaces
   an existing baseline and therefore changes the future comparison reference.

The three records form an evidence chain: scripts produce detailed run data;
the solver STORY document interprets and preserves the engineering conclusion;
perf-gates protects an accepted, fixed workload from later regressions. A
single passing test, a single timing table, or a baseline update alone is not
sufficient evidence for a performance claim.

### 10. Stabilizing An Optimization On An Existing Solver

An optimization is promoted to the stable solver path only after this sequence
has completed for the affected solver and supported routes:

1. **Release correctness matrix.** Run the solver's relevant Release tests,
   including applicable ignored correctness/parity stories. Require numerical
   parity, expected discrete work and lifecycle behavior; investigate failures
   before interpreting any timing.
2. **Compare existing performance gates.** Run `perf-gates compare` for the
   affected solver/scenarios against the current compatible baseline. Require
   no confirmed regression. An unstable measurement is inconclusive and must
   be repeated or diagnosed, not treated as either a pass or a regression.
3. **Run the solver's release story/benchmark matrix.** Use its script to
   collect compact `tabled` reports in `test_reports/`. Compare the new
   measurements with historical STORY entries and prior reports, including
   full-solve wall time and the stage/counter evidence that explains the
   result. Record the conclusion and relevant numerical measurements in the
   solver's STORY document; do not promote an optimization based only on a
   faster isolated callback when full-solve impact is immaterial or unknown.
4. **Refresh accepted performance baselines.** Only after correctness and
   historical-result review accepts the new behavior, run `perf-gates record
   --force` for the affected scenarios and inspect the generated baseline/run
   metadata. This intentionally makes the accepted result the reference for
   subsequent comparisons; it must not be used to hide an unexplained
   regression. Keep unrelated solver baselines unchanged.

If a solver has no compatible baseline yet, baseline creation is a separate
calibration decision; it does not count as a successful regression comparison.
The source revision, machine, toolchain, workload and gate identity must be
appropriate for the intended reference. Exploratory or unstable results stay
in run reports and do not advance the stable baseline.

### 11. Review Checklist

Before declaring a solver production-ready, check that:

- telemetry is disabled cheaply and enabled with documented semantics;
- logging is off by default, and every enabled severity writes a discoverable
  text log;
- ported algorithms have a versioned local reference and an auditable parity
  ledger for matched counter semantics and intentional deviations;
- counters and timing scopes are covered by tests;
- all important routes have correctness and parity evidence;
- compact tables are written to reviewable files;
- technical logs are separated from reports;
- long campaigns continue after individual failures;
- partial, empty and interrupted campaigns are visible as such;
- repeated statistical baselines exist for claims about performance;
- release evidence is archived and reproducible;
- Sequential/Parallel/Auto and whole/chunked routes have matched evidence when
  the solver supports them;
- callback-only and full-solve policy results are not conflated;
- worker, chunk and applicability telemetry distinguishes zero from unavailable;
- the solver documentation explains known workload-dependent tradeoffs.

This guideline is an engineering standard, not a promise that every solver
must expose every possible backend. Unsupported routes should be represented by
typed, explicit results and documented as unsupported rather than silently
falling back to a different algorithm.
