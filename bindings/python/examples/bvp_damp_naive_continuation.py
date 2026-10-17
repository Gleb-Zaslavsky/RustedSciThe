"""Prepared parameter continuation using BVP_Damp's native Naive route.

The scalar ODE y' = rate with y(0) = 0 has the exact profile y(x) = rate*x.
This example keeps the fixed uniform mesh and symbolic preparation while
changing only the numeric parameter.
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
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="dense",
        telemetry="timings",
    ),
)

solver.prepare()
for rate in (1.0, 2.0, 0.5):
    solver.set_parameters([rate])
    solution = solver.solve_prepared()
    print(
        f"rate={rate:g}, y(1)={solution.values()[0][-1]:.8f}, "
        f"iterations={solution.telemetry_counters().get('iterations', 0)}"
    )
