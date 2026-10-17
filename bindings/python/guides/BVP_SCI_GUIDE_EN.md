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
| Parameter series | `parameters`, `parameter_values`, `set_parameters(...)` | Rebind numeric values without changing equation structure |
| Restart | `restart(mesh, initial_state)` | Replaces discretization and numerical initial iterate |

For Banded, configure `SolverOptions.set_banded_width(lower, upper)` with the
collocation Jacobian's actual half-bandwidth. Do not guess a narrow band and
interpret a failed solve as evidence against the layout.

Structured layouts need Jacobian information compatible with the selected
route. Inspect `capabilities("bvp_sci")` and reject unsupported combinations
explicitly rather than assuming a callback fallback works for every layout.
The optional SuperLU route is an opt-in sparse factorization backend; it is
not required by the base Python wheel. If unavailable, inspect the reported
fallback/provenance before treating timings as SuperLU evidence.

## Parameters, restart and singular terms

The direct wrapper supports parameter binding, `set_parameters`, restart with
a new mesh/initial guess, mesh refinement controls, dense output, and SciPy-
style singular-term inputs `(singular_term_a, singular_term_matrix)`. A
parameter change does not imply a new symbolic model. Restart resets the
numerical trajectory state while retaining the prepared route where supported.
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

The Python wrapper currently has a separate callback path for numerical
boundary residuals/Jacobians; consult `capabilities` and the constructor docs
for its narrower route. The full symbolic wrapper is the normal choice when
equations are available as `Expr`.

See `examples/bvp_sci_sparse.py` for the basic route and
`examples/bvp_sci_frontend_layouts.py` for a matched exact linear BVP matrix.
`examples/bvp_sci_aot.py` compares Lambdify and AOT on the same Sparse,
AtomNative problem and requires `tcc` or `gcc` on `PATH`.
Rust/Python parity and performance claims belong to the dedicated core test
and benchmark reports, not to example output.
