# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 193.896300 | 172.032500 | 0.203000 | 0.290100 | 2.652400 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 652.999300 | 630.784800 | 0.557400 | 0.348700 | 2.526400 | 1.000000e+00 |
