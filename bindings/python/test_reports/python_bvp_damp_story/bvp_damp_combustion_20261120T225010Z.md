# BVP_Damp Python combustion story

Native verified combustion fixture; initial 1000-node Sparse mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | sparse | 1001 | 4 | 8 | 1 | 1 | 6851.093700 | 6844.786800 | 2.315500 | 1.806900 | 1.462700 | 1.001471e+00 |
| atom_native | sparse | 1001 | 4 | 8 | 1 | 1 | 6738.634100 | 6732.203500 | 3.040200 | 1.247800 | 1.468800 | 1.001471e+00 |
