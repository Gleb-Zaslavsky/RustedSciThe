# Python Bindings Architectural Guideline

This document defines the contract for the Python distribution of
RustedSciThe. The Python package is a user-facing interface to the Rust crate,
not a reduced solver edition or a second numerical implementation. Python
convenience must not silently remove capabilities that the corresponding Rust
API supports.

The Rust solver API and [core architectural guideline](../../src/RST_ARCHITECTURAL_GUIDELINE.md)
remain authoritative for numerical behavior, algorithms and native telemetry.
This document adds Python-boundary requirements: API fidelity, cross-platform
delivery, interop, and measurement of wrapper costs.

## 1. Preserve Native Capability

Python must expose the meaningful public capabilities of each selected Rust
solver. In particular, every symbolic frontend, callback execution route,
linear algebra backend/layout, execution policy, lifecycle operation and
solver-specific setting supported by that Rust API must remain available from
Python.

- Do not replace a solver's native builder or configuration surface with a
  smaller generic options object that loses controls.
- Keep independent axes independent. For example, `ExprLegacy`/`AtomNative`
  select symbolic representation, while `Lambdify`/`AOT` select callback
  execution; neither is a linear algebra backend.
- Keep solver-specific APIs distinct when their mathematical contracts differ.
  A shared Python facade may improve discoverability but must not erase
  method-specific options or lifecycle semantics.
- Mirror native defaults. If Python intentionally chooses a different default,
  document the reason and test that behavior explicitly.
- Validate unsupported combinations early and raise a typed, contextual Python
  exception. Never silently substitute another frontend, backend, policy or
  solver mode.
- Treat an unavailable optional dependency, such as the SuperLU bridge, as an
  explicit capability state. Report whether it was compiled in and whether it
  can be loaded; do not present a fallback result as if the requested route
  had run.

Maintain a route-by-route contract matrix covering native Rust support, Python
constructor/configuration, runtime forwarding, result and error behavior,
`.pyi` declarations, guides and tests. A capability is either exposed and
tested, or recorded as intentionally unavailable with a concrete native/API
reason. “Not yet wrapped” is a tracked gap, not an implicit design choice.

## 2. Test The Supported Cross-Product

Solver tests must exercise supported combinations across the axes that affect
behavior, rather than merely smoke-test one default constructor. The matrix
should include, where applicable:

- each symbolic frontend and callback execution route;
- each Dense, Sparse or Banded backend/layout supported by that solver;
- sequential, parallel and automatic policies;
- analytic and finite-difference callback paths;
- fresh, prepared, warm, restart and continuation lifecycles;
- solver-specific settings, defaults, typed failures and telemetry modes.

Use exact solutions, invariants, independent solver comparisons or established
Rust story-test workloads as correctness gates. Test unsupported combinations
too, asserting that errors retain solver, route, stage and native error kind.
Large matrices must be broad enough to catch cross-axis omissions, but can use
small deterministic workloads in debug CI and representative medium/large
workloads in a slower release campaign. Do not turn benchmark timing thresholds
into portable correctness assertions.

Keep Python-boundary tests distinct from Rust-core tests: a core test does not
measure Python conversion or wrapper behavior. Validate both editable builds
and the exact installed wheel before release; once Linux wheels are supported,
run the same contract suite on Windows and Linux.

## 3. Windows And Linux Are Product Platforms

The intended distribution supports Windows and Linux. Platform support is not
complete merely because Rust code compiles on both operating systems or an
sdist exists. Each platform needs an actual wheel and an installed-artifact
contract run.

- Build the Windows and Linux wheels from the same immutable release source
  and versioned sdist.
- Test clean installs outside the source checkout on both platforms, with
  supported Python versions, NumPy imports, native-extension loading and the
  public API suite.
- Exercise platform-sensitive behavior: shared-library discovery, optional
  native dependencies, subprocesses, paths, compiler discovery and animation
  worker startup/failure.
- Keep optional tools optional. Preflight reports must identify missing tools
  without making unrelated solver imports or solves fail.
- Do not claim a platform or interpreter range until its installed wheel has
  passed the corresponding gate. Record artifact hashes and test environment.

Platform-specific implementation details may differ, but public Python
semantics and numerical contracts must not.

## 4. Preserve Native Telemetry And Logging

When a Rust solver has telemetry, Python must provide a way to select the
native telemetry mode and retrieve its structured snapshot. When Rust has
logging, Python must expose the ability to enable or disable it and pass its
configuration through to Rust. Do not invent telemetry or logging for a solver
that does not have it, and do not silently drop a capability that does.

- Preserve the native mode semantics exactly. In particular, do not alias
  `Counters`, `Timings` and `Detailed` merely because a simpler Python API is
  convenient.
- Keep telemetry separate from logs. Telemetry is structured in-memory
  measurements; logs are human-readable persistent text output configured and
  written on the Rust side.
- Preserve lifecycle scopes and stage names. A missing measurement must stay
  distinguishable from a measured zero; do not fill absent values with `0`.
- Return telemetry as stable Python mappings or typed result objects, document
  units and scope, and include the corresponding fields in `.pyi` files.
- Keep solver logging disabled by default when that is the Rust default. If
  enabled, ensure the Rust logger writes to its configured persistent file;
  Python should not capture and relay the log as a substitute.
- Test mode forwarding, defaults, disabled overhead paths, scope consistency,
  and preservation of native diagnostics on errors.

## 5. Measure Python-Boundary Cost

Python benchmarks measure the wrapper boundary, not the numerical solver as
such. Rust-core benchmarks and story matrices are authoritative for algorithm,
linear-algebra backend, frontend and solver-performance comparisons. Do not
duplicate those matrices in Python or use a Python wall-clock ranking to claim
that one native solver route is faster than another.

The Python benchmark's question is: what additional cost or behavior does the
Python interface introduce around a native operation? Measure and report,
where relevant:

- Python facade construction/validation and the PyO3 call boundary;
- NumPy input/output conversion and materialization;
- Python-owned callback crossings and GIL-bound callback cost;
- Python-observed wall time for a matched call or lifecycle;
- fresh versus prepared/warm/continuation wrapper-call overhead.

Native preparation and solver timings/counters may be included as attribution
context and correctness/provenance checks. They are not a second native-core
benchmark. A signed `Python wall - native solver timing` value is an
**attribution residual**, not pure Python CPU time: it can include PyO3
conversion, native orchestration outside solver timers, and measurement noise.
Preserve its sign, name its scope, and never add it to native stages as though
the intervals were disjoint. If a benchmark includes native AOT compilation,
label that lifecycle cost explicitly; use core benchmarks for claims about
generated-code performance or solver speed.

Use matched equations, tolerances, initial states, outputs and lifecycle scopes
when comparing Python call routes. State clearly when rows are not
apple-to-apple (for example, Python callbacks versus native symbolic
callbacks). Use repeated samples and report a robust summary such as
median/MAD, plus correctness and relevant native counters. Boundary
performance claims must be measured using a clean environment with the exact
installed release wheel, outside the checkout import path. Every report must
record Python/package versions and the resolved Python-package and native
extension paths so an editable build cannot be mistaken for a wheel.

## 6. NumPy Is The Numerical Interop Standard

Use NumPy arrays as the standard Python boundary representation for numerical
vectors, matrices, grids and trajectory data. This is the expected
interoperability surface for scientific Python users.

- Accept ordinary array-like input where the API promises convenience, then
  normalize once at the boundary to the documented dtype, dimensionality,
  shape and memory requirements.
- Prefer `float64` for current numerical APIs unless a native solver explicitly
  supports another dtype. Reject invalid dimensions and non-finite values with
  contextual exceptions rather than allowing opaque downstream failures.
- Return detached NumPy arrays with documented axis ordering and ownership;
  never expose a view into mutable Rust solver workspace.
- Avoid hidden repeated conversions inside callbacks or iterations. Report
  conversion cost separately in boundary benchmarks and document when a copy is
  required.
- Keep scalar metadata, enums, statuses and structured diagnostics as ordinary
  Python values; NumPy is for numerical arrays, not every return type.
- Test array shape, dtype, ownership/lifetime, non-contiguous input handling
  and round-trip behavior for public array APIs.

## Definition Of Done For A Binding Change

A Python-facing feature is ready when its native capability and semantics are
understood, the wrapper forwards all relevant options without narrowing them,
errors and diagnostics retain context, NumPy and `.pyi` contracts agree, and
tests cover supported and rejected routes. User documentation and examples
must explain non-obvious choices. Performance-sensitive changes also need a
matched boundary measurement. Release-facing changes must pass clean installed
wheel gates on every claimed platform.

The active implementation queue and release status are tracked in
[`TODO.md`](TODO.md); this guideline is the normative design contract, not a
duplicate task list.
