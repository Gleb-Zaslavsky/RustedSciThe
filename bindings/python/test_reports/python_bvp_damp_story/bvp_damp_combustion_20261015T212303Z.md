# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 216.120300 | 192.423300 | 0.177700 | 0.479000 | 2.450400 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 675.809900 | 651.779500 | 0.555500 | 0.443700 | 2.414200 | 1.000000e+00 |
