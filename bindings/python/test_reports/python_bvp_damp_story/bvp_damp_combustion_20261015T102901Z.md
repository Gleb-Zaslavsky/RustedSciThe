# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 212.926100 | 187.751100 | 0.234900 | 0.489400 | 3.377100 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 667.175200 | 642.631200 | 0.563800 | 0.435500 | 3.006900 | 1.000000e+00 |
