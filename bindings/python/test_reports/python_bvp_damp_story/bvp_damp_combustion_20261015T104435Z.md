# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 204.648000 | 180.778400 | 0.181100 | 0.467100 | 2.542500 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 657.502400 | 632.677700 | 0.600400 | 0.430800 | 2.416400 | 1.000000e+00 |
