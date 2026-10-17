# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 211.192300 | 187.231000 | 0.173600 | 0.367900 | 2.442600 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 680.193700 | 656.035900 | 0.579500 | 0.454000 | 2.380900 | 1.000000e+00 |
