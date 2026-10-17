# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 226.641500 | 201.202200 | 0.227900 | 0.506200 | 2.826500 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 710.141800 | 684.847700 | 0.571400 | 0.525800 | 2.782100 | 1.000000e+00 |
