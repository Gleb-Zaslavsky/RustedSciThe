# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 216.847700 | 193.003300 | 0.187100 | 0.385300 | 2.529100 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 676.346700 | 651.557800 | 0.572300 | 0.435200 | 2.550300 | 1.000000e+00 |
