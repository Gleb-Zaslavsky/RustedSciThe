# Nonlinear Python Boundary: Lambdify vs Cold AOT

Runtime: Python=3.14.6, rustedscithe=0.1.4, package_path=`C:\Users\user\AppData\Local\Temp\rustedscithe-014-boundary-clean\Lib\site-packages\rustedscithe\__init__.py`, native_extension=`C:\Users\user\AppData\Local\Temp\rustedscithe-014-boundary-clean\Lib\site-packages\rustedscithe\_native.pyd`.
Matched dense systems: `x_i^2 - p_i = 0`, with exact positive roots.
Times are median/MAD in ms. `prepare_ms` includes a fresh AOT lowering,
compile and link for AOT rows (`RebuildAlways`, fresh artifact directory).
Each prepared model then performs a warm parameter-rebind solve series.
Lambdify preparation is symbolic preparation only. Solve telemetry is
nested inside Python wall time and must not be added to it.
Counters and exact-root checks are included to make route mismatches visible.
This measures the installed wrapper path, not the Rust core alone.
Warm-series length: 8; full lifecycle includes constructor, preparation, first solve and all warm calls.
Rust solve time sums native stages for the initial and warm solves; it remains nested in Python wall time.
`wall - Rust` values are signed attribution residuals, not pure Python time; they include PyO3 conversion/dispatch and native work outside solver timers.
Break-even is a median-based estimate: extra AOT preparation divided by per-call warm-solve savings; `not reached` means AOT warm calls were not faster.

| n | frontend | execution | constructor ms | prepare ms | first solve ms | first wall - Rust ms | warm series ms | warm wall - Rust ms | avg warm call ms | full lifecycle ms | estimated AOT break-even rebinds | Rust solve ms | residual evals | Jacobian evals | linear solves | max root error |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | expr_legacy | lambdify | 0.012/0.001 | 0.255/0.059 | 0.019/0.003 | 0.010/0.002 | 0.091/0.012 | 0.065/0.007 | 0.011/0.002 | 0.415/0.099 | n/a | 0.035/0.005 | 37 | 37 | 28 | 1.130e-13 |
| 4 | expr_legacy | aot | 0.021/0.001 | 241.017/3.258 | 0.067/0.002 | 0.023/0.002 | 0.136/0.030 | 0.095/0.034 | 0.017/0.004 | 241.155/3.371 | not reached | 0.081/0.005 | 37 | 37 | 28 | 1.130e-13 |
| 4 | atom_native | lambdify | 0.003/0.001 | 0.017/0.000 | 0.005/0.000 | 0.001/0.000 | 0.059/0.002 | 0.038/0.002 | 0.007/0.000 | 0.084/0.003 | n/a | 0.024/0.000 | 37 | 37 | 28 | 1.130e-13 |
| 4 | atom_native | aot | 0.019/0.000 | 221.315/1.382 | 0.038/0.000 | 0.018/0.000 | 0.073/0.002 | 0.053/0.003 | 0.009/0.000 | 221.442/1.357 | not reached | 0.041/0.001 | 37 | 37 | 28 | 1.130e-13 |
| 16 | expr_legacy | lambdify | 0.018/0.004 | 0.208/0.050 | 0.015/0.002 | 0.003/0.001 | 0.100/0.002 | 0.058/0.002 | 0.012/0.000 | 0.343/0.066 | n/a | 0.054/0.002 | 38 | 38 | 29 | 2.220e-16 |
| 16 | expr_legacy | aot | 0.029/0.001 | 225.563/1.203 | 0.074/0.012 | 0.023/0.001 | 0.160/0.009 | 0.095/0.007 | 0.020/0.001 | 225.840/1.203 | not reached | 0.117/0.006 | 38 | 38 | 29 | 2.220e-16 |
| 16 | atom_native | lambdify | 0.010/0.000 | 0.060/0.001 | 0.012/0.000 | 0.002/0.000 | 0.107/0.001 | 0.051/0.001 | 0.013/0.000 | 0.189/0.002 | n/a | 0.067/0.000 | 38 | 38 | 29 | 2.220e-16 |
| 16 | atom_native | aot | 0.030/0.002 | 235.113/4.148 | 0.054/0.003 | 0.019/0.001 | 0.112/0.001 | 0.067/0.001 | 0.014/0.000 | 235.324/4.133 | not reached | 0.081/0.003 | 38 | 38 | 29 | 2.220e-16 |
| 32 | expr_legacy | lambdify | 0.028/0.006 | 0.239/0.033 | 0.029/0.002 | 0.004/0.001 | 0.192/0.004 | 0.075/0.002 | 0.024/0.001 | 0.487/0.050 | n/a | 0.142/0.006 | 38 | 38 | 29 | 3.864e-14 |
| 32 | expr_legacy | aot | 0.037/0.001 | 231.371/2.160 | 0.120/0.011 | 0.024/0.003 | 0.317/0.083 | 0.139/0.028 | 0.040/0.010 | 231.697/2.125 | not reached | 0.254/0.045 | 38 | 38 | 29 | 3.864e-14 |
| 32 | atom_native | lambdify | 0.017/0.000 | 0.115/0.002 | 0.030/0.001 | 0.003/0.000 | 0.212/0.002 | 0.071/0.001 | 0.027/0.000 | 0.375/0.005 | n/a | 0.167/0.001 | 38 | 38 | 29 | 3.864e-14 |
| 32 | atom_native | aot | 0.033/0.001 | 225.617/5.310 | 0.072/0.002 | 0.018/0.000 | 0.195/0.000 | 0.083/0.000 | 0.024/0.000 | 226.137/5.089 | 105498 | 0.166/0.002 | 38 | 38 | 29 | 3.864e-14 |
