# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 209.994700 | 186.478700 | 0.181700 | 0.385100 | 2.366500 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 662.760200 | 638.090300 | 0.578400 | 0.547400 | 2.395400 | 1.000000e+00 |
