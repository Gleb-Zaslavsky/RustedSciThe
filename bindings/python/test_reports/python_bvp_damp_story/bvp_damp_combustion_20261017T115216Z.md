# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 198.885200 | 177.133600 | 0.182400 | 0.282200 | 2.577800 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 654.375100 | 632.579500 | 0.527400 | 0.332600 | 2.429900 | 1.000000e+00 |
