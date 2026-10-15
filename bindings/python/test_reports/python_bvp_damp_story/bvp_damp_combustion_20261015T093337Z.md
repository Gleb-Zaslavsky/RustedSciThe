# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 202.674400 | 178.944300 | 0.182900 | 0.484300 | 2.556600 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 662.798600 | 638.752200 | 0.563000 | 0.466500 | 2.372500 | 1.000000e+00 |
