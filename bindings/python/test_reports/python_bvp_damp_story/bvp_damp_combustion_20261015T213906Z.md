# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 208.611800 | 183.180900 | 0.203200 | 0.402500 | 2.505500 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 674.472900 | 649.281600 | 0.557000 | 0.440300 | 2.608400 | 1.000000e+00 |
