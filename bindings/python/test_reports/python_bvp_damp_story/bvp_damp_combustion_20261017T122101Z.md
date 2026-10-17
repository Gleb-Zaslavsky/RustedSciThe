# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 204.439700 | 182.731400 | 0.159600 | 0.275300 | 2.362300 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 675.038600 | 652.432600 | 0.570100 | 0.363900 | 2.469300 | 1.000000e+00 |
