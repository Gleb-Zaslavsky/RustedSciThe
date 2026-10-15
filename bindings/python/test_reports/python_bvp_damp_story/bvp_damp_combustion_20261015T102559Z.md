# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 211.336200 | 187.342000 | 0.194200 | 0.383000 | 2.460700 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 662.035100 | 638.009700 | 0.565000 | 0.443400 | 2.389100 | 1.000000e+00 |
