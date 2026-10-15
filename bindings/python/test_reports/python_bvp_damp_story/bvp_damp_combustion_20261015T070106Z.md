# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 522.194600 | 489.754500 | 0.465700 | 0.687700 | 4.013600 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 1256.433800 | 1229.323700 | 0.835000 | 0.634800 | 2.659000 | 1.000000e+00 |
