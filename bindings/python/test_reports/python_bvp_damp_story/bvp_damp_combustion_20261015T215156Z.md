# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 207.386400 | 182.839600 | 0.196700 | 0.478400 | 2.674500 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 700.120200 | 675.285400 | 0.623600 | 0.479200 | 2.510100 | 1.000000e+00 |
