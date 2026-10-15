# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 223.256500 | 198.909400 | 0.219700 | 0.368400 | 2.505000 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 680.221100 | 655.387000 | 0.617100 | 0.487100 | 2.406500 | 1.000000e+00 |
