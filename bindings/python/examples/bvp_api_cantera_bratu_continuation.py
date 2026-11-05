"""Lower-branch Bratu continuation: u'' + lambda*exp(u) = 0.

Symmetry reduces u(0)=u(1)=0 to u(0)=0, v(0.5)=0 with v=u'. This supplies
one endpoint condition per state as required by Cantera. Ten lambda values
from 0.3 to 3.0 stay away from the turning point. This is parameter stepping,
not pseudo-arclength continuation; it does not trace the fold/upper branch.

Run: python bindings/python/examples/bvp_api_cantera_bratu_continuation.py --steps 256
"""

import math

import numpy as np

from rustedscithe import Expr

from _cantera_continuation import run


def check_exact(lam, mesh, values):
    # lambda = 8*a^2/cosh(a)^2 is monotone on [0, 1], bracketing this schedule.
    low, high = 0.0, 1.0
    for _ in range(60):
        a = 0.5 * (low + high)
        if 8.0*a*a / math.cosh(a)**2 < lam:
            low = a
        else:
            high = a
    a = 0.5 * (low + high)
    z = a * (2.0*mesh - 1.0)
    exact = np.column_stack((2.0*np.log(math.cosh(a) / np.cosh(z)), -4.0*a*np.tanh(z)))
    error = np.max(np.abs(values - exact))
    h = mesh[1] - mesh[0]
    # Discretization error is distinct from the Newton correction tolerance.
    assert error < 20.0*h*h + 1e-6, f"Bratu analytic-profile error: {error:.3e}"
    assert values[-1, 0] > 0.0
    print(f"  analytic lower-branch max error={error:.3e}")


def main():
    run(
        names=["u", "v"],
        expressions=[Expr.parse("v"), Expr.parse("-lambda*exp(u)")],
        right=0.5,
        boundaries=[(0, "left", 0.0), (1, "right", 0.0)],
        initial=[0.0, 0.0],
        parameter="lambda",
        parameters=[[0.3*i] for i in range(1, 11)],
        rhs=lambda lam, y: [y[1], -lam*math.exp(y[0])],
        check=check_exact,
    )


if __name__ == "__main__":
    main()

