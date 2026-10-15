# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 202.503600 | 178.649800 | 0.188000 | 0.467900 | 2.468100 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 672.225400 | 647.411800 | 0.740500 | 0.487300 | 2.498700 | 1.000000e+00 |
