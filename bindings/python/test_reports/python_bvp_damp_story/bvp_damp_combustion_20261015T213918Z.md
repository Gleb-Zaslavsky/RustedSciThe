# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 208.002900 | 182.932400 | 0.202900 | 0.377300 | 2.742900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 672.329900 | 648.134500 | 0.577400 | 0.478700 | 2.431200 | 1.000000e+00 |
