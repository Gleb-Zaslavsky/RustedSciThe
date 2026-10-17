# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 212.667400 | 190.328000 | 0.231500 | 0.308800 | 2.727800 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 665.253800 | 643.432100 | 0.521600 | 0.342900 | 2.376100 | 1.000000e+00 |
