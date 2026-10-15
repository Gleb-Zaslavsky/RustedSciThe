# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 226.694600 | 202.632200 | 0.171000 | 0.471600 | 2.558500 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 667.048600 | 642.462400 | 0.575800 | 0.461600 | 2.412300 | 1.000000e+00 |
