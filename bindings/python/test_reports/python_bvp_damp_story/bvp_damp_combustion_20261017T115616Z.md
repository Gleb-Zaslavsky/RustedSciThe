# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 212.695300 | 190.234000 | 0.193900 | 0.302100 | 2.597900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 681.310200 | 657.621700 | 0.661600 | 0.422300 | 3.051900 | 1.000000e+00 |
