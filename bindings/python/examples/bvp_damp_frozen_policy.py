"""Frozen-Jacobian Newton and fixed-parameter rebinding.

The scalar problem y' = rate with y(0) = 0 has the exact solution
y(x) = rate*x. The mesh and prepared symbolic problem are reused while a
fixed numeric input changes. The prior profile seeds each subsequent solve;
``every_m`` refreshes the Jacobian periodically.
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
    # Recompute the frozen Jacobian every two solves. This is a visible policy
    # example, not a claim that periodic reuse is best for every BVP.
    policy="every_m",
    policy_parameters=[2.0],
    options=SolverOptions(
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="dense",
        # The example reports Jacobian rebuild counts as well as timings.
        telemetry="detailed",
    ),
)

solver.prepare()
for rate in (1.0, 2.0, 0.5):
    solution = solver.solve_with_parameter_values([rate])
    endpoint = solution.values()[0][-1]
    counters = solution.telemetry_counters()
    if abs(endpoint - rate) > 1e-8:
        raise AssertionError(f"rate={rate:g}: expected y(1)={rate:g}, got {endpoint:g}")
    print(
        f"rate={rate:g}, y(1)={endpoint:.8f}, "
        f"policy={solver.provenance()['frozen_policy']}, "
        f"jacobians={counters['jacobian_recalculations']}"
    )
