# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 497.409700 | 475.617600 | 0.222100 | 0.305300 | 2.659100 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 935.582300 | 913.871400 | 0.524300 | 0.349600 | 2.442100 | 1.000000e+00 |
