"""Solve a scalar root and a small least-squares problem.

The root equation ``x**2 - 4 = 0`` has roots +/-2; the initial guess is
positive, so Newton should converge to +2. The fitting residual ``x - 2``
has the exact minimizer 2. These two calls demonstrate distinct public
problem types and their detached NumPy result/telemetry methods.
"""

from rustedscithe import Expr, SymbolicSystem, create_fit, create_nonlinear


system = SymbolicSystem([Expr.parse("x^2 - 4")], ["x"])
nonlinear = create_nonlinear(system, method="newton", telemetry="timings")
nonlinear.prepare()
root = nonlinear.solve([1.0])
print("root:", root.solution_array())
print("Newton timings:", root.telemetry_timings_ms())

fit = create_fit([Expr.parse("x - 2")], [0.0], telemetry="counters")
fit_result = fit.solve()
print("fit:", fit_result.solution_array())
print("fit counters:", fit_result.telemetry_counters())
