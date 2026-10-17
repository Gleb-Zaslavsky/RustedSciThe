# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 194.041000 | 172.377900 | 0.170700 | 0.252100 | 2.537400 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 657.893900 | 635.776900 | 0.552100 | 0.339500 | 2.514200 | 1.000000e+00 |
