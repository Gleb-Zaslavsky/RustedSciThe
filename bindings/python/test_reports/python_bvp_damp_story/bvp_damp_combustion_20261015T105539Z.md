# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 201.986700 | 177.747800 | 0.177200 | 0.350300 | 2.413300 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 675.774300 | 650.923200 | 0.571500 | 0.452800 | 2.535300 | 1.000000e+00 |
