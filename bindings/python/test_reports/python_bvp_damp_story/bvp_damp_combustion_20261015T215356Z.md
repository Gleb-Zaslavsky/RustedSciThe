# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 224.371500 | 195.153600 | 0.233500 | 0.423500 | 3.020000 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 709.255900 | 684.888600 | 0.555800 | 0.499900 | 2.433200 | 1.000000e+00 |
