"""Fit once, then evaluate, extrapolate, and reconstruct internal gaps."""

from rustedscithe import UniversalFitting


solver = UniversalFitting("lm")
solver.with_data([0.0, 1.0, 2.0, 4.0, 5.0, 6.0], [1.0, 3.5, 6.0, 11.0, 13.5, 16.0])
solver.with_arg("x")
solver.with_equation_str("a*x+b")
solver.with_unknowns(["a", "b"])
solver.with_initial_guess([1.0, 0.0])
fit = solver.build()

x, values = fit.evaluate_uniform_range(2.0, 4.0, 5)
print("interpolated", x, values[:, 0])

x, values = fit.extrapolate_uniform_range(6.5, 8.5, 3)
print("right extrapolation", x, values[:, 0])

gaps = fit.fill_missing_intervals([(2.0, 4.0, 3)])
gap_x, gap_values = gaps[0]
print("restored gap", gap_x, gap_values[:, 0])
