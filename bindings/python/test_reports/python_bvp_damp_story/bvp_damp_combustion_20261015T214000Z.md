# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 215.064400 | 190.557200 | 0.207400 | 0.468200 | 2.458400 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 703.249100 | 679.008700 | 0.548100 | 0.466100 | 2.437300 | 1.000000e+00 |
