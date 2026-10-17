# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 198.461300 | 177.183900 | 0.166000 | 0.275100 | 2.374600 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 648.915300 | 627.206000 | 0.547700 | 0.328300 | 2.378600 | 1.000000e+00 |
