# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 200.458500 | 178.737300 | 0.202300 | 0.300300 | 2.606200 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 650.217400 | 628.035200 | 0.540600 | 0.326900 | 2.656300 | 1.000000e+00 |
