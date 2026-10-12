# BDF User Guide

The standalone BDF solver is a dense, SciPy-faithful implicit solver for
small-to-medium fully coupled systems. It is intentionally not a second
structured solver. For sparse or banded storage, use LSODE2 and select its
`Sparse` or `Banded` matrix route explicitly.

## Choosing A Backend

The public BDF options support three practical execution styles:

- native callbacks when the right-hand side and Jacobian already exist as Rust
  code;
- symbolic `Lambdify` callbacks for a prepared symbolic problem;
- generated AOT callbacks for a deployable compiled artifact.

For symbolic assembly, `IvpSymbolicAssemblyBackend::ExprLegacy` preserves the
legacy expression route and `IvpSymbolicAssemblyBackend::AtomView` uses the
native symbolic view. The two routes have the same numerical contract. Their
preparation cost and callback cost are workload-dependent, so choose using
measurements rather than a universal ranking.

For application code, the stable convenience imports are available from
`RustedSciThe::numerical::BDF::prelude`:

```rust
use RustedSciThe::numerical::BDF::prelude::{
    BdfSolverOptions, BdfStatus, BdfTelemetryMode, IvpSymbolicAssemblyBackend,
    ODEsolver,
};
```

Low-level linear backend traits are intentionally not part of the prelude;
import them from `BDF_solver` only when supplying a custom dense factorization.

## Native Dense Solve

`examples/bdf_native_dense_guide.rs` is the smallest native example:

```text
cargo run --no-default-features --example bdf_native_dense_guide
```

It installs a native residual and a constant dense analytic Jacobian, enables
counter telemetry, uses typed `try_solve`, and checks the result against the
closed-form solution.

## Symbolic Parameter Continuation

`examples/bdf_symbolic_continuation_guide.rs` demonstrates AtomView plus
Lambdify. A prepared residual/Jacobian is reused while only the parameter value
and numerical BDF history are changed:

```text
cargo run --no-default-features --example bdf_symbolic_continuation_guide
```

Use continuation when the symbolic structure is unchanged. A changed equation
shape, state dimension, Jacobian layout, or backend policy requires a new
prepared assembly.

For a new initial state or a new interval with the same prepared model, use
`try_restart_with_initial_state(t0, y0, t_bound)`. It resets BDF history,
initial Jacobian and factorization, but does not repeat symbolic preparation or
AOT publication. The fallible API rejects non-finite and dimension-mismatched
states before changing the prepared model.

## AOT And Toolchains

`examples/bdf_aot_guide.rs` uses AtomView and generated C/tcc callbacks:

```text
cargo run --no-default-features --example bdf_aot_guide
```

The example exits successfully with a diagnostic message when `tcc` is not on
`PATH`. AOT preparation includes materialization, build, link and publication;
warm continuation should reuse the published artifact instead of rebuilding it.
The shared AOT lifecycle driver also provides an explicit per-attempt toolchain
deadline. A timed-out compiler is terminated and reported as a typed timeout;
the legacy blocking execution method remains available for compatibility.

When timing telemetry is enabled, `solver.aot_provenance()` returns the typed
lifecycle identity together with the immutable telemetry snapshot. It records
the build policy, codegen backend, compiler override, artifact keys, cache
hits/misses, build/link attempts and runtime publication counters in one
matched object. Use this rather than combining fields from separate solver
instances or benchmark scopes.

## Telemetry And Errors

Telemetry is off by default. `BdfTelemetryMode::Counters` is a low-overhead
choice for work counts; `BdfTelemetryMode::Timings` adds diagnostic stage
timers. Nested timing scopes are inclusive and must not be added together.

Prefer the typed `try_*` API (`try_solve`, continuation and configuration
methods) in applications. It reports invalid dimensions, missing callbacks,
unsupported backend combinations and preparation failures as typed errors.
Runtime step failures are also classified: a rejected Newton linear solve,
Newton iteration-budget exhaustion, malformed/non-finite callback output, and
the high-level maximum-step budget are distinct error categories. The checked
path does not use panic-based validation for these conditions; legacy methods
such as `solve` and `generate` remain compatibility adapters that may panic on
an error.

Scalar `rtol` must be finite and positive; scalar `atol` must be finite and
non-negative. Vector tolerances follow the same rules and must have one entry
per state component. `try_solve` retains the initial point and every accepted
point in the result, including a partial trajectory when a typed failure
occurs. It does not promise to catch a panic raised by user code inside a
callback.

## Scope Boundary

The standalone solver stores dense Jacobians and uses dense linear algebra.
Do not pass a dense `jac_sparsity` promise expecting sparse execution. For large
diffusion or banded systems, LSODE2 is the supported route and provides the
structured matrix backends and their corresponding telemetry and story tests.

## Numerical-Core Parity With SciPy

The numerical controller was checked against the pinned official SciPy 1.18.1
BDF implementation in `src/numerical/ivp_scipy_reference`. The reference adds
observational telemetry only; it does not change SciPy's formulas or controller
branches. The matched Rust/Python story covers ten workloads: constant and
finite-difference Jacobians, a deliberately stale constant Jacobian and retry,
Robertson, backward integration, automatic first-step selection, HIRES,
combustion-like kinetics, a non-autonomous mixed-scale system, and a fully
coupled dense system.

The comparison checks final states, three common interior times, solver state
metadata, and mapped work counters. The maximum scaled final-state drift was
`1.109e-7`; the maximum scaled interior-sample drift was `0.405`, below the
explicit acceptance threshold of `2`. Accepted/rejected steps, order changes,
Newton work, and linear solves matched in the tested cases. The release timing
benchmark separately compares SciPy construction/integration/state capture
with Rust numerical initialization/integration/result collection; symbolic
preparation is excluded, so its timings are not full-API or universal speed
claims.

The remaining counter differences have specific explanations, not unexplained
algorithm changes. SciPy's `nfev` excludes finite-difference Jacobian probes;
Rust counts actual RHS callbacks and reuses the initial RHS as the initial FD
baseline. In the matched FD case this gives SciPy `nfev=506` and Rust `nfev=509`
(510 raw SciPy RHS callbacks). A deliberately stale-Jacobian case has one
additional Rust LU (`245` vs `244`): at the final bound, floating-point
rounding makes Rust clip a candidate just beyond the bound and invalidate LU,
while SciPy reaches the bound and reuses it. The preceding per-step LU counts
match, and the trajectory remains within the parity envelope.

This is numerical-core parity evidence for the covered cases, not bitwise
identity or complete SciPy API parity. Rust's configurable `max_steps`, typed
callback/error contracts, and no-progress/underflow behavior are Rust API
features with independent tests. Rust BDF currently has no public dense-output
evaluator: interior values were checked against separate Rust solves at the
same times, not against a Rust interpolation implementation. These are explicit
scope boundaries rather than unresolved controller mismatches. The full
post-fix release matrix should still be rerun before treating the broader BDF
release campaign as completely green.
