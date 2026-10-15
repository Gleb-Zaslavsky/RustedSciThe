# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 203.828900 | 179.575200 | 0.182000 | 0.372000 | 2.499900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 667.488500 | 643.468600 | 0.554300 | 0.423800 | 2.395500 | 1.000000e+00 |
