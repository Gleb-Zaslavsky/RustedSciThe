# BVP_sci Guide for Python

BVP_sci is the collocation boundary-value solver whose numerical method
mirrors SciPy's `solve_bvp` family. The Rust implementation adds the crate's
symbolic frontends, execution routes, matrix layouts, telemetry and typed
errors. Boundary conditions are residual equations, not a state-to-observation
dictionary.

## Problem shape and minimal solve

For an order-one problem `y'=-y`, prescribe one left boundary residual. The
initial mesh and state values are guesses; they are not boundary conditions.

```python
import numpy as np
from rustedscithe import Expr, SolverOptions, solve_bvp

y = Expr.variable("y")
x = np.linspace(0.0, 1.0, 12)
guess = np.exp(-x)  # node-major: (nodes, state)
solution = solve_bvp(
    "bvp_sci", [-y], ["y"], [Expr.parse("ya_y - 1")], x, guess,
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="sparse", telemetry="timings",
    ),
    output="dense",
)
print(solution.sample(1.0), "exact:", np.exp(-1.0))
```

The solution exposes the refined mesh and a node-major `(nodes, states)`
NumPy array. Use `sample(x)` for the continuous collocation output rather than
assuming every requested point survives mesh refinement.

## Frontends, layouts and AOT

Select `expr_legacy` or `atom_native` independently from `lambdify` or `aot`.
The matrix layout may be Dense, Sparse or Banded. These are actual solver
routes, not conversions between storage formats. Dense can be appropriate for
small fully coupled systems; Sparse/Banded may win when the global collocation
Jacobian has exploitable structure. No layout is universally fastest.

The direct Python constructor keeps the numerical controls visible rather
than hiding them behind a lowest-common-denominator options object:

| Control | Python keyword | Meaning |
| --- | --- | --- |
| Newton tolerance | `tolerance`, `boundary_tolerance` | Interior and boundary residual criteria |
| Work limits | `max_nodes`, `max_newton_iterations`, `max_mesh_refinements`, `max_backtracking_steps` | Explicit solver budgets |
| Output | `output="final"`, `"mesh"`, `"dense"` | Retain final values, mesh values or continuous interpolant |
| Boundary Jacobian | `boundary_jacobian="symbolic"` or `"finite_difference"` | Applies to symbolic boundary expressions |
| Singular term | `singular_term_a`, `singular_term_matrix` | Supplied as a pair; `a` is the left endpoint |
| Free parameter unknowns | `parameters`, `parameter_guess`, `set_parameter_guess(...)` | Newton initial guesses for unknowns solved jointly with the state profile; `parameter_values` is a legacy constructor alias |
| Restart | `restart(mesh, initial_state)` | Replaces discretization and numerical initial iterate |

For symbolic ExprLegacy/AtomView routes, Banded infers the state-Jacobian
structure automatically; no width is needed. `set_banded_width(lower, upper)`
remains an optional structural hint for advanced callers and must not truncate
the true symbolic pattern. Numerical callbacks have no symbolic structure to
inspect, so automatic inference is unavailable there.

Structured layouts need Jacobian information compatible with the selected
route. Inspect `capabilities("bvp_sci")` and reject unsupported combinations
explicitly rather than assuming a callback fallback works for every layout.
The optional SuperLU route is an opt-in sparse factorization backend; it is
not required by the base Python wheel. If unavailable, inspect the reported
fallback/provenance before treating timings as SuperLU evidence.

## Parameters, restart and singular terms

In BVP_sci, declared `parameters` are free unknowns, not fixed model inputs.
The solver determines them together with the state profile and requires the
boundary residual dimension to include one extra condition per parameter.
`parameter_guess` (or `set_parameter_guess(...)`) sets only the initial Newton
guess for those unknowns; `parameter_guess()` inspects the current guess vector.
The constructor keyword `parameter_values` remains a deprecated alias and
cannot be supplied together with `parameter_guess`.
After a successful solve, the existing mesh and
profile are already retained for the next solve. This is distinct from fixed
parameter rebinding in BVP_Damp. `restart(mesh, initial_state)` explicitly
replaces the numerical discretization and state guess while retaining symbolic
preparation where supported; it does not change the parameter schema.

### Fixed external-parameter loops

Use `external_parameters=[...]` for fixed model inputs that appear in the RHS
or symbolic boundary expressions but are not Newton unknowns. Set their
initial values with `external_parameter_values=[...]`; the callback argument
vector is ordered as free parameters first, then external parameters. The
external values do not add Jacobian columns or boundary conditions.

`solve_parameter_loop([[...], [...]])` solves each external-input vector in
order using the same prepared symbolic plan. Each `BvpSciParameterLoopPoint`
contains its input vector, either a detached solution or a typed native error,
and a telemetry snapshot taken after that point. These snapshots are cumulative
since plan preparation, not per-point deltas. An ordinary solve failure does not abort
the loop: the next point starts from the last successful mesh, profile and
free-parameter solution (or the original initial state if none succeeded).
Only a failure to restore that checkpoint stops the loop; inspect
`report.stopped_early` and call `report.raise_if_stopped()` when that must be
fatal. Numerical RHS callback routes currently reject named external inputs.
The example `examples/bvp_sci_external_parameter_loop.py` demonstrates a
success/failure/success sequence.
Choose boundary residuals and singular terms from the mathematical model, not
to force a particular linear layout.

## Telemetry and errors

Select `telemetry="off"`, `"counters"`, `"timings"` or `"detailed"` through
`SolverOptions`. Counters measure discrete work without clock reads; timings
measure durations without discrete counters; detailed combines both and keeps
per-mesh/Newton diagnostics. Inspect preparation, collocation, factorization,
solve and refinement scopes separately where reported. Timing scopes can overlap. The result also reports selected layout,
frontend/execution provenance and sparse factorization policy. Invalid
boundary shapes, non-finite data, unsupported combinations and numerical
failures use typed Python exceptions.

### Direct numerical RHS callbacks

When the RHS is already implemented numerically, BVP_sci exposes the native
Dense callback route without pretending that it used symbolic preparation.
Pass empty `expressions`, select both `frontend="numerical"` and
`execution="numerical"`, and provide `rhs_callback(x, y, parameters)`. The
callback returns one derivative per state. `rhs_jacobian_callback` may return
the dense `df/dy` matrix and `rhs_parameter_jacobian_callback` may return
`df/dp`; either omitted derivative uses the native finite-difference fallback.
The collocation boundary conditions may remain `Expr` residuals or use the
separate boundary callbacks. This route currently supports Dense only and
does not support AOT. Python callbacks execute under the GIL, so keep them
small; the numerical collocation and linear solve remain in Rust.

```python
from rustedscithe import BvpSciSolver, Expr, SolverOptions

solver = BvpSciSolver(
    [], ["y"], [Expr.parse("ya_y - 1")],
    [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
    options=SolverOptions(
        frontend="numerical", execution="numerical",
        matrix_layout="dense", telemetry="detailed",
    ),
    rhs_callback=lambda _x, y, _p: [-y[0]],
    rhs_jacobian_callback=lambda _x, _y, _p: [[-1.0]],
    output="dense",
)
solution = solver.solve()
print(solution.sample(1.0))
```

`capabilities("bvp_sci")` advertises this callback route and its Dense-only
analytical-Jacobian contract. Use the symbolic wrapper whenever an `Expr` RHS
is available so Rust can prepare exact derivatives and select other layouts.

See `examples/bvp_sci_unknown_parameter_guess.py` for the free-parameter
unknown/initial-guess lifecycle. `examples/bvp_sci_sparse.py` shows the basic route and
`examples/bvp_sci_frontend_layouts.py` for a matched exact linear BVP matrix.
`examples/bvp_sci_aot.py` compares Lambdify and AOT on the same Sparse,
AtomNative problem and requires `tcc` or `gcc` on `PATH`.
`examples/bvp_sci_numerical_callbacks.py` demonstrates the Dense callback
route with an analytical Jacobian.
Rust/Python parity and performance claims belong to the dedicated core test
and benchmark reports, not to example output.
