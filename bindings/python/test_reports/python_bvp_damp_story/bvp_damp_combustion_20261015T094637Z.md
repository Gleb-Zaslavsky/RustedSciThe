# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 217.211100 | 193.394100 | 0.182400 | 0.496100 | 2.456000 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 673.361000 | 649.404100 | 0.563500 | 0.469700 | 2.373900 | 1.000000e+00 |
