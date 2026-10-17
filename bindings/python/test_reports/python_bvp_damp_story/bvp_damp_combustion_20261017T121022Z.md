# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 190.672400 | 169.378600 | 0.160200 | 0.278400 | 2.504900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 644.920500 | 623.129000 | 0.542800 | 0.348600 | 2.479100 | 1.000000e+00 |
