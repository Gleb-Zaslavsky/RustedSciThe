# BVP_Damp Python combustion story

Native verified combustion fixture; initial 1000-node Sparse mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | sparse | 1001 | 4 | 8 | 1 | 1 | 8631.652000 | 8438.401600 | 16.782800 | 5.321400 | 45.391800 | 1.001471e+00 |
| atom_native | sparse | 1001 | 4 | 8 | 1 | 1 | 8326.936700 | 8126.811800 | 29.254700 | 4.314900 | 41.337000 | 1.001471e+00 |
