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

Telemetry must be:

- disabled by default or available through an explicit low-overhead mode;
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

### 2. Every Solver Has A Broad Evidence Corpus

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

### 3. Compact Tables Are The Primary Human-Readable Output

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

### 4. Reports And Technical Logs Are Separate Artifacts

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

### 5. Long Campaigns Must Be Non-Fail-Fast

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

### 6. Compact Release Evidence And Statistical Baselines Are Different

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

### 7. Review Checklist

Before declaring a solver production-ready, check that:

- telemetry is disabled cheaply and enabled with documented semantics;
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
