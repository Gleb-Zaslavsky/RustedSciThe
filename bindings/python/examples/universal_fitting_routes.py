"""Choose LM, Gavin or VarPro through the distinct UniversalFitting facade.

This example fits the same through-origin line through LM and Gavin,
then uses VarPro for an exponential with a linear offset.  The universal
facade delegates route selection to Rust; it is not a replacement for the
standalone classes when route-specific covariance or detailed telemetry is
needed.  Unsupported option combinations raise typed errors instead of being
silently ignored.
"""

import math

from rustedscithe import UniversalFitting


x_data = [index * 0.25 for index in range(12)]
line_data = [1.7 * x_value for x_value in x_data]


def fit_line(method: str) -> None:
    # UniversalFitting is configured in steps because it selects a method
    # before the shared data/equation contract is assembled by build().
    fit = UniversalFitting(method)
    fit.with_data(x_data, line_data)
    fit.with_arg("x")
    fit.with_equations_str(["slope*x"])
    fit.with_unknowns(["slope"])
    fit.with_initial_guess([1.0])
    fit.with_frontend("atom_native")
    fit.with_telemetry("detailed")
    if method == "gavin":
        fit.with_gavin_update_policy("nielsen")
    result = fit.build()
    parameters = result.solution_map()
    assert parameters is not None
    print(f"{method}: {parameters}; R^2={result.r_squared}")
    assert abs(parameters["slope"] - 1.7) < 1e-5


fit_line("lm")
fit_line("gavin")

# VarPro solves the linear coefficients internally for each nonlinear shape
# parameter. It deliberately has a narrower contract: ExprLegacy + sequential.
decay_x = [index * 0.2 for index in range(30)]
decay_y = [2.5 * math.exp(-value / 1.7) + 0.8 for value in decay_x]
varpro = UniversalFitting("varpro")
varpro.with_data(decay_x, decay_y)
varpro.with_arg("x")
varpro.with_unknowns(["tau"])
varpro.with_initial_guess([1.2])
varpro.with_basis_str("tau", "exp(-x/tau)")
varpro.with_invariant_str("1")
varpro.with_telemetry("detailed")
result = varpro.build()
print("varpro:", result.solution_map(), "R^2:", result.r_squared)

varpro_parameters = result.solution_map()
assert varpro_parameters is not None
assert abs(varpro_parameters["tau"] - 1.7) < 0.05
assert abs(varpro_parameters["c0"] - 2.5) < 0.05
assert abs(varpro_parameters["c1"] - 0.8) < 0.05
