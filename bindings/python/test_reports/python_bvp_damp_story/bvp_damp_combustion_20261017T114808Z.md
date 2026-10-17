# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 204.136100 | 181.822100 | 0.165100 | 0.262600 | 2.559800 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 667.505700 | 645.621900 | 0.532500 | 0.366600 | 2.447000 | 1.000000e+00 |
