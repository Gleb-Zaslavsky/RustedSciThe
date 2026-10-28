"""Fixed-parameter re-solving using BVP_Damp's native Naive route.

The scalar ODE y' = rate with y(0) = 0 has the exact profile y(x) = rate*x.
This example keeps the fixed uniform mesh and symbolic preparation while
changing only a fixed numeric input. Each solve after the first is warm-started
from the previous profile; ``rate`` is not a Newton unknown.
"""

from rustedscithe import BvpDampNaiveSolver, Expr, SolverOptions


solver = BvpDampNaiveSolver(
    expressions=[Expr.parse("rate")],
    state_variables=["y"],
    boundary_conditions={"y": [(0, 0.0)]},
    mesh=[0.0, 0.25, 0.5, 0.75, 1.0],
    initial_state=[0.0] * 5,
    parameters=["rate"],
    parameter_values=[1.0],
    options=SolverOptions(
        # AtomNative is used to demonstrate prepared symbolic rebinding; the
        # fixed-mesh Naive route keeps this example focused on rebinding.
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="dense",
        # Both iteration counters and stage timings are printed below.
        telemetry="detailed",
    ),
)

# Build the symbolic plan once; reuse the last profile as the next Newton guess.
solver.prepare()
for rate in (1.0, 2.0, 0.5):
    solution = solver.solve_with_parameter_values([rate])
    endpoint = solution.values()[0][-1]
    counters = solution.telemetry_counters()
    if abs(endpoint - rate) > 1e-8:
        raise AssertionError(f"rate={rate:g}: expected y(1)={rate:g}, got {endpoint:g}")
    print(
        f"rate={rate:g}, y(1)={endpoint:.8f}, "
        f"iterations={counters['iterations']}"
    )
