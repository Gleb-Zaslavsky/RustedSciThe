# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 209.812600 | 185.455800 | 0.189800 | 0.365500 | 2.434500 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 686.058100 | 660.746300 | 0.592500 | 0.491400 | 2.636600 | 1.000000e+00 |
