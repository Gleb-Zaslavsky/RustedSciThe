# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 237.416200 | 209.732400 | 0.224300 | 0.508500 | 2.765700 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 708.800800 | 683.623000 | 0.659500 | 0.461500 | 2.450800 | 1.000000e+00 |
