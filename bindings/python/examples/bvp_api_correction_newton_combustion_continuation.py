"""Synthetic six-state combustion-like BVP, NOT a physical kinetics model.

Adapted from BVP_CorrectionNewton/tests/large_workloads.rs's six-state fixture. State
names are mathematical labels, not calibrated temperature, heat flux, species
fractions, or conserved physical fluxes. `heat_scale` multiplies one nonlinear
source; no CorrectionNewton chemistry library or real mechanism is used.

Run: python bindings/python/examples/bvp_api_correction_newton_combustion_continuation.py --steps 256
"""

import math

from rustedscithe import Expr

from _correction_newton_continuation import run


def rhs(scale, y):
    teta, q, c0, j0, c1, j1 = y
    return [
        q, -0.5*q + 0.2*scale*math.exp(teta)*c0,
        j0, -0.3*j0 - 0.5*c0 + 0.1*teta,
        j1, -0.2*j1 - 0.3*c1 + 0.05*c0*c0,
    ]


def check_profile(_scale, _mesh, values):
    assert values[-1, 0] < -0.01, "nontrivial Teta profile"
    assert values[-1, 2] > 1.01, "nontrivial C0 profile"
    assert values[-1, 4] < -0.001, "nontrivial C1 profile"


def main():
    names = ["Teta", "q", "C0", "J0", "C1", "J1"]
    equations = [
        "q", "-0.5*q + 0.2*heat_scale*exp(Teta)*C0",
        "J0", "-0.3*J0 - 0.5*C0 + 0.1*Teta",
        "J1", "-0.2*J1 - 0.3*C1 + 0.05*C0^2",
    ]
    run(
        names=names,
        expressions=[Expr.parse(eq) for eq in equations],
        right=1.0,
        boundaries=[
            (0, "left", 0.0), (1, "right", 0.0),
            (2, "left", 1.0), (3, "right", 0.0),
            (4, "left", 0.0), (5, "right", 0.0),
        ],
        initial=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        parameter="heat_scale",
        parameters=[[0.8 + 0.05*i] for i in range(10)],
        rhs=rhs,
        check=check_profile,
    )


if __name__ == "__main__":
    main()

