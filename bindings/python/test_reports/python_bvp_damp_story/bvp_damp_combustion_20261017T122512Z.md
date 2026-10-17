# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 210.443800 | 188.844300 | 0.179000 | 0.286900 | 2.405300 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 698.983200 | 676.948200 | 0.555800 | 0.392100 | 2.442900 | 1.000000e+00 |
