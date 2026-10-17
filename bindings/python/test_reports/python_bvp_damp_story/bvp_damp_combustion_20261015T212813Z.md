# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 225.989900 | 202.351200 | 0.178500 | 0.386700 | 2.421100 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 693.419400 | 668.271800 | 0.572900 | 0.493800 | 2.409000 | 1.000000e+00 |
