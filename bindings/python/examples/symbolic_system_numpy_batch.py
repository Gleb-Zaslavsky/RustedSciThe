"""Prepare and evaluate a vector residual/Jacobian system with NumPy buffers.

Both frontends run inside Rust. AtomNative selects the core's packed AtomView
path directly; this example does not convert an Expr Jacobian after the fact.
"""

import numpy as np

from rustedscithe import Expr, SymbolicFrontend, SymbolicSystem


def main() -> None:
    x, y = Expr.symbols(["x", "y"])
    system = SymbolicSystem([x**2 + y, x * y], ["x", "y"])
    prepared = system.lambdify(frontend=SymbolicFrontend.AtomNative, diagnostics=True)

    # The sample axis comes first. A strided view is accepted without callers
    # having to normalize it to a contiguous copy.
    padded_points = np.array([[2.0, -1.0, 3.0], [4.0, -1.0, 5.0]])
    points = padded_points[:, ::2]
    residuals = prepared.evaluate_batch(points)
    jacobians = prepared.evaluate_jacobian_batch(points)

    # Reuse application-owned arrays in repeated evaluations. These methods
    # return the same array object they filled, so they compose with pipelines.
    prepared.evaluate_batch_into(points, residuals)
    prepared.evaluate_jacobian_batch_into(points, jacobians)

    print("variables:", prepared.variables())
    print("residuals (samples, equations):\n", residuals)
    print("Jacobians (samples, equations, variables):\n", jacobians)
    print("preparation route:", prepared.provenance())
    print("native diagnostics:", prepared.diagnostics())


if __name__ == "__main__":
    main()
