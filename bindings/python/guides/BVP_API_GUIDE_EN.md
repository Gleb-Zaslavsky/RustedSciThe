# Typed BVP API

`BvpSolverRequest` is an application-facing selector over the Rust `BVP_api`
facade. It keeps solver selection out of physics/model code while forwarding
the selected route to Rust. It does not replace the direct
`BvpDampSolver`/`BvpSciSolver` APIs when route-specific diagnostics are needed.

```python
import math

from rustedscithe import BvpSolverRequest, Expr, SolverOptions

y = Expr.variable("y")
mesh = [i / 20 for i in range(21)]
guess = [[math.exp(-x)] for x in mesh]  # [node][state], including both ends

request = BvpSolverRequest.sci(
    expressions=[-y],
    state_variables=["y"],
    independent_variable="x",
    mesh=mesh,
    boundary_conditions=[(0, "left", 1.0)],
    initial_guess=guess,
    options=SolverOptions(
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="banded",  # bandwidth is inferred from symbolic structure
        telemetry="detailed",
    ),
    tolerance=1e-5,
)
solver = request.prepare()
solution = solver.solve()
print(solution.values_array()[-1, 0])
print(solver.telemetry_snapshot())
```

## Routes And Shapes

Factories are tagged by native solver family: `damped`, `frozen`, or `sci`.
`mesh` must be finite and strictly increasing. Boundary conditions are
`(state_index, "left" | "right", value)` triples; the indices refer to the
ordered `state_variables` list. Equations and state variables must have equal
length, with one scalar endpoint condition per state.

Guesses are Python/NumPy-friendly node-major rows. Damped Newton expects one
row per mesh interval because its native iterate is interval-based. Frozen
and BVP_sci expect one row per mesh node. Damped and Frozen currently require
a uniform mesh; the native constructors accept endpoints and a count rather
than arbitrary coordinates. BVP_sci accepts a nonuniform mesh.

The shared `SolverOptions` carries common frontend, execution, matrix layout,
callback policy, telemetry, logging and AOT settings. Each factory also
exposes its algorithm-specific controls. Unsupported choices raise a typed
Python exception with `stage`, `solver`, `route` and `native_kind`; the facade
does not silently fall back. The generic facade supports symbolic Lambdify
and AOT routes. Numerical Python callback routes remain on the direct
`BvpSciSolver` API.

For BVP_sci free parameters, provide all of `free_parameter_names`,
`free_parameter_guess` and `boundary_residual`. The residual callback receives
`(ya, yb, parameters)` and returns one extra boundary residual per unknown.
An optional `boundary_jacobian` returns `(dR/dya, dR/dyb, dR/dp)` as nested
row-major lists. These callbacks execute on the Python side and therefore
cross the GIL; use the direct BVP_sci symbolic-boundary API if its expression
route better fits the problem.

BVP_sci fixed model inputs are a separate facility: provide
`external_parameters` and, optionally, their initial
`external_parameter_values` when creating the request. Their symbolic
callbacks are prepared once. `solve_parameter_loop([[...], ...])` then sets
each new value vector before solving and seeds the next attempt from the last
successful profile. Failed points remain in the route-native report and do not
abort later points unless checkpoint recovery itself fails. This is distinct
from `free_parameter_names`, whose values are Newton unknowns. Damped/Frozen
loops use their existing `parameters`/`parameter_values` request inputs.

The returned report is intentionally route-specific:
`BvpSciParameterLoopReport` for BVP_sci and
`BvpDampParameterLoopReport` for Damped/Frozen. Point telemetry and typed
failures are not normalized into a lossy common schema. See
`examples/bvp_api_parameter_loop.py` for a runnable BVP_sci example.

`BvpApiSolution` always provides detached node-major values and the accepted
mesh. `sci_details` additionally exposes the complete native BVP_sci result,
including status and requested dense output. `telemetry_snapshot()` returns
route-native counters, timings and scope metadata. A missing map entry means
the measurement was not collected; measured zero values are retained. Rust
logs are still written by the native logger, not returned as Python data.

Use `BvpPostprocessPlan` with `execute_postprocessing()` for native CSV/text,
Markdown, Plotters/Gnuplot or terminal output. The returned rows report each
action's `done`, `skipped` or `failed` status. BVP_sci dense-output sampling is
available through `solution.sci_details`; the shared postprocessing dataset
contains stored mesh values.

Run the paired example with `python examples/bvp_api_solver_selection.py`.
