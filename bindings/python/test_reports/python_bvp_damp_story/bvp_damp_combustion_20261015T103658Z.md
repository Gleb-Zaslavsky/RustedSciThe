# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 206.383900 | 182.309500 | 0.195800 | 0.347600 | 2.455900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 659.232200 | 634.577900 | 0.559400 | 0.419000 | 2.394200 | 1.000000e+00 |
