# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 212.698000 | 188.169000 | 0.201600 | 0.372700 | 2.479000 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 662.779500 | 638.656600 | 0.553400 | 0.472900 | 2.391000 | 1.000000e+00 |
