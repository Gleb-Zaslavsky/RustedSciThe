# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 217.371300 | 193.067600 | 0.203300 | 0.398000 | 2.569200 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 670.741700 | 646.500500 | 0.580400 | 0.465200 | 2.365800 | 1.000000e+00 |
