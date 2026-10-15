# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 201.629100 | 177.878700 | 0.178100 | 0.342500 | 2.453000 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 661.034100 | 637.149000 | 0.534300 | 0.396200 | 2.361600 | 1.000000e+00 |
