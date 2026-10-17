"""Frozen-Jacobian Newton with a periodic refresh policy.

The scalar problem y' = rate with y(0) = 0 has the exact solution
y(x) = rate*x. The mesh and prepared symbolic problem are reused while the
numeric parameter changes; ``every_m`` refreshes the Jacobian periodically.
"""

from rustedscithe import BvpDampFrozenSolver, Expr, SolverOptions


solver = BvpDampFrozenSolver(
    expressions=[Expr.parse("rate")],
    state_variables=["y"],
    boundary_conditions={"y": [(0, 0.0)]},
    mesh=[0.0, 0.25, 0.5, 0.75, 1.0],
    initial_state=[0.0] * 5,
    parameters=["rate"],
    parameter_values=[1.0],
    policy="every_m",
    policy_parameters=[2.0],
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
        f"policy={solver.provenance()['frozen_policy']}, "
        f"jacobians={solution.telemetry_counters().get('jacobian_recalculations', 0)}"
    )
