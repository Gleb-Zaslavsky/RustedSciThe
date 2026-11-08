# BVP_Damp Python combustion story

Native verified combustion fixture; initial 1000-node Sparse mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | sparse | 1001 | 4 | 8 | 1 | 1 | 7605.805700 | 7452.533300 | 8.510900 | 4.639900 | 41.273800 | 1.001471e+00 |
| atom_native | sparse | 1001 | 4 | 8 | 1 | 1 | 7201.680400 | 7052.461500 | 17.820700 | 3.296600 | 33.770100 | 1.001471e+00 |
