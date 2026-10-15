# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 202.483900 | 177.940400 | 0.203300 | 0.374000 | 2.479900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 668.236400 | 644.113300 | 0.585000 | 0.449700 | 2.390000 | 1.000000e+00 |
