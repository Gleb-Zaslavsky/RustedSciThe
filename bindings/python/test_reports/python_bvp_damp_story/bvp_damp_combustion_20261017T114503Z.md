# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 197.501600 | 175.322700 | 0.210900 | 0.308600 | 3.021400 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 648.997700 | 627.150900 | 0.530400 | 0.331600 | 2.421700 | 1.000000e+00 |
