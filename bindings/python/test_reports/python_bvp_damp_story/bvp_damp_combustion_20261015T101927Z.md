# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 203.621900 | 179.886700 | 0.191100 | 0.381100 | 2.384100 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 675.290000 | 650.387600 | 0.574900 | 0.469500 | 2.412900 | 1.000000e+00 |
