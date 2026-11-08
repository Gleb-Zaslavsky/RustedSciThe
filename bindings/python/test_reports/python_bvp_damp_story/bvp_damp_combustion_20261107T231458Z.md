# BVP_Damp Python combustion story

Native verified combustion fixture; initial 1000-node Sparse mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | sparse | 1001 | 4 | 8 | 1 | 1 | 8725.200900 | 8581.117500 | 11.134700 | 3.641000 | 33.758800 | 1.001471e+00 |
| atom_native | sparse | 1001 | 4 | 8 | 1 | 1 | 8717.993400 | 8485.953500 | 30.818600 | 4.901100 | 50.639700 | 1.001471e+00 |
