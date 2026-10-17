# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 208.823400 | 185.027400 | 0.177400 | 0.380100 | 2.427600 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 678.056800 | 653.396300 | 0.583700 | 0.550400 | 2.374400 | 1.000000e+00 |
