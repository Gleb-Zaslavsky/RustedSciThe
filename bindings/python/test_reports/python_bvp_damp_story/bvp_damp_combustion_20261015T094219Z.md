# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 207.619200 | 183.929100 | 0.185100 | 0.352100 | 2.483600 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 661.602000 | 637.722400 | 0.561600 | 0.430700 | 2.361000 | 1.000000e+00 |
