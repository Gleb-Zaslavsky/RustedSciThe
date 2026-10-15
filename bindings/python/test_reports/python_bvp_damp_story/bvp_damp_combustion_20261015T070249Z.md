# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 266.109300 | 238.995200 | 0.328500 | 0.782600 | 2.874900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 818.664400 | 792.678200 | 0.663100 | 0.552400 | 2.587200 | 1.000000e+00 |
