# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 207.439100 | 185.010200 | 0.370100 | 0.291300 | 2.477700 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 665.548500 | 643.726300 | 0.545900 | 0.331400 | 2.449500 | 1.000000e+00 |
