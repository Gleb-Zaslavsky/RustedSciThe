# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 201.869900 | 176.935100 | 0.204000 | 0.356400 | 2.715900 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 664.685500 | 640.450100 | 0.545800 | 0.442700 | 2.561100 | 1.000000e+00 |
