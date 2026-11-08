# BVP_Damp Python combustion story

Native verified combustion fixture; initial 1000-node Sparse mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | sparse | 1001 | 4 | 8 | 1 | 1 | 7352.800600 | 7213.145700 | 8.119900 | 3.729400 | 33.667900 | 1.001471e+00 |
| atom_native | sparse | 1001 | 4 | 8 | 1 | 1 | 6990.375500 | 6841.621600 | 17.871200 | 3.250000 | 33.951600 | 1.001471e+00 |
