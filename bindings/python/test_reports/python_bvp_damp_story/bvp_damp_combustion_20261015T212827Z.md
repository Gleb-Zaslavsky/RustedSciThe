# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 207.252000 | 183.689400 | 0.169400 | 0.456100 | 2.406700 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 673.842200 | 649.153700 | 0.561900 | 0.457100 | 2.420600 | 1.000000e+00 |
