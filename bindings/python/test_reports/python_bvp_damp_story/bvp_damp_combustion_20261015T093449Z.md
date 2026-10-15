# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 215.348200 | 191.388700 | 0.186400 | 0.487400 | 2.615400 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 676.472900 | 652.362600 | 0.555500 | 0.444700 | 2.421500 | 1.000000e+00 |
