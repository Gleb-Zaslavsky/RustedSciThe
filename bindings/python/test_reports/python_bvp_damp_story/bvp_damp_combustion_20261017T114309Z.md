# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 194.788800 | 173.295000 | 0.156200 | 0.393000 | 2.539600 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 649.815600 | 627.836900 | 0.531700 | 0.341000 | 2.541500 | 1.000000e+00 |
