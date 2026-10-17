# BVP_Damp Python combustion story

Native verified combustion fixture; initial 20-node mesh; Lambdify only.
This is an explicit slow story, separate from the fast Python contract suite.

| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expr_legacy | dense | 21 | 8 | 20 | 3 | 3 | 219.904700 | 195.570600 | 0.186000 | 0.374700 | 2.501200 | 1.000000e+00 |
| atom_native | dense | 21 | 8 | 20 | 3 | 3 | 688.611600 | 664.209100 | 0.563800 | 0.444100 | 2.452000 | 1.000000e+00 |
