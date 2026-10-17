# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 212.566400 | 187.465300 | 0.290500 | 0.365500 | 3.129300 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 727.632100 | 705.475300 | 0.600500 | 0.353900 | 2.412200 | 1.000000e+00 |
