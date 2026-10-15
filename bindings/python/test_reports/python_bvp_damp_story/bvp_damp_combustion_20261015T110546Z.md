# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 220.145600 | 196.089300 | 0.191900 | 0.390300 | 2.457900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 701.794400 | 677.096600 | 0.575900 | 0.427900 | 2.424800 | 1.000000e+00 |
