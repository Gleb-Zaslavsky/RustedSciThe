# BVP_CorrectionNewton User Guide

`BVP_CorrectionNewton` is a fixed-uniform-mesh boundary-value route implementing a
damped correction-Newton numerical core. Its nonlinear iteration is
independent of BVP_Damp and BVP_sci. The public client facade is the shared
[`BVP_api`](../BVP_api/BVP_API_GUIDE_EN.md); this page documents the route's
capabilities and boundaries.

The numerical iteration was inspired by Cantera's `MultiNewton` approach, while
this implementation has its own Rust API, symbolic frontends, linear backends,
telemetry, and validation contract.

## Numerical Scope

The discretization is a fixed-grid trapezoid system. Boundary-constrained
unknowns are eliminated from the packed Newton state and restored when the
profile is returned. There is no mesh refinement, pseudo-transient recovery,
or mesh restart in this solver route. Fixed external inputs are inferred from
symbolic expressions and may be bound/rebound without rebuilding prepared
Lambdify/AOT callbacks or the Jacobian pattern. `solve_parameter_loop` runs an
ordered fixed-input continuation: the first point uses the configured initial
guess, later points use the most recent successful profile, and failed points
remain typed per-point outcomes without aborting later attempts. The report
includes full profiles, the physical mesh, and native partial telemetry. See
the facade examples `examples/bvp_api_correction_newton_combustion_continuation.rs` and
`examples/bvp_api_correction_newton_bratu_continuation.rs`.

The Newton controls include the component-scaled weighted RMS correction,
fraction-to-boundary bounds, CorrectionNewton's fixed `sqrt(2)` damping reduction,
Jacobian age/reuse, and its bounded Jacobian-refresh retries after exhausted
damping. A successful result is published only after the configured
correction criterion is met. Failure returns a typed error and the last
accepted state/partial report; rejected trials are not published as a result.

## Symbolic And Linear Routes

`CorrectionNewtonOptions` selects `ExprLegacy` or native `AtomView`, and either
Lambdify or AOT callback execution. AOT requires an explicit
`SymbolicIvpGeneratedBackendConfig` through `with_aot_config`; preparation or
runtime failures preserve typed causes and do not fall back silently.

Dense, Sparse, and `BandedAuto` are supported. Sparse and Banded assemble the
global Jacobian directly from a fixed coordinate pattern; they do not build a
dense global matrix. Banded widths are inferred from that symbolic pattern,
not supplied by the caller. Dense is intended for compact systems and
cross-checks; large workloads should use Sparse or Banded when structurally
appropriate.

Sparse and Banded solve results are checked for finiteness and backward
residual using reusable solver-owned scratch. An invalid numerical result is
returned as a typed `NonFiniteLinearSolution` or
`LinearResidualCheckFailed`, preserving the partial solve report; it is never
accepted as a Newton step. Dense retains its backend singularity check.

## Diagnostics

Logging defaults to `SolverLogLevel::Off` and uses the shared crate logger.
Native telemetry defaults to Off; Counters does not read clocks, while Timings
collects both counters and durations. A failed numerical solve retains the
partial report in its typed error. Preparation telemetry belongs to the
symbolic/AOT lifecycle and is not folded into native solve timings.

See `examples/bvp_api_correction_newton.rs` for the facade entry point. The focused
correctness/report matrix is `scripts/bvp_correction_newton_release_matrix.ps1`; the
large workload regression groups are `bvp-correction-newton-matrix` and the separately
selectable ten-point `bvp-correction-newton-parameter-continuation` group in
`scripts/bvp_correction_newton_perf_gates.ps1`.
