"""Solve a scalar root and a small least-squares problem.

The root equation ``x**2 - 4 = 0`` has roots +/-2; the initial guess is
positive, so Newton should converge to +2. The fitting residual ``x - 2``
has the exact minimizer 2. These two calls demonstrate distinct public
problem types and their detached NumPy result/telemetry methods.
"""

from rustedscithe import Expr, SymbolicSystem, create_fit, create_nonlinear


system = SymbolicSystem([Expr.parse("x^2 - 4")], ["x"])
# Root finding and fitting share symbolic expressions, but have different
# numerical contracts: the first solves equations, the second minimizes data.
nonlinear = create_nonlinear(system, method="newton", telemetry="timings")
nonlinear.prepare()
root = nonlinear.solve([1.0])
root_value = float(root.solution_array()[0])
if abs(root_value - 2.0) > 1e-6:
    raise AssertionError(f"Newton converged to an unexpected root: {root_value}")
print("root:", root.solution_array())
print("Newton timings:", root.telemetry_timings_ms())

# The fit residual is already an equation in the unknown x; its target is 0.
fit = create_fit([Expr.parse("x - 2")], [0.0], telemetry="counters")
fit_result = fit.solve()
fit_value = float(fit_result.solution_array()[0])
if abs(fit_value - 2.0) > 1e-8:
    raise AssertionError(f"least-squares minimizer is incorrect: {fit_value}")
print("fit:", fit_result.solution_array())
print("fit counters:", fit_result.telemetry_counters())
