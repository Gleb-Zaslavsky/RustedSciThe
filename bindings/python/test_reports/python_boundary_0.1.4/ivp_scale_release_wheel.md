# Scaled Python BDF Boundary Workloads

Runtime: Python=3.14.6, rustedscithe=0.1.4, package_path=`C:\Users\user\AppData\Local\Temp\rustedscithe-014-boundary-clean\Lib\site-packages\rustedscithe\__init__.py`, native_extension=`C:\Users\user\AppData\Local\Temp\rustedscithe-014-boundary-clean\Lib\site-packages\rustedscithe\_native.pyd`.
Matched diagonal decay IVP, `n=1,8,32,128`, end time=0.2, repeats=9.
The exact endpoint is `exp(-rate_i*t_end)` for each component. Times are median/MAD ms.
Rust stage timings and counters are nested in Python solve wall time; do not add them together.
`wall - Rust solve` is a signed attribution residual, not pure Python time; it can include PyO3 conversion/dispatch and native work outside solver timers.
The Python callback route crosses the GIL during numerical evaluation and is not a matched symbolic frontend claim.

| n | route | Expr build | NumPy input | constructor | prepare wall | solve wall | Rust solve | wall - Rust solve | residual | Jacobian | factorization | residual calls | Jacobian calls | linear solves | LU count | NumPy output | max exact error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | expr_legacy | 0.001/0.000 | 0.000/0.000 | 0.003/0.000 | 0.007/0.000 | 0.053/0.001 | 0.046/0.001 | 0.007/0.000 | 0.015/0.000 | 0.000/0.000 | 0.001/0.000 | 46 | 1 | 44 | 7 | 0.000/0.000 | 2.741e-08 |
| 1 | atom_native | 0.001/0.000 | 0.000/0.000 | 0.003/0.000 | 0.011/0.000 | 0.051/0.000 | 0.043/0.000 | 0.008/0.000 | 0.013/0.000 | 0.000/0.000 | 0.001/0.000 | 46 | 1 | 44 | 7 | 0.000/0.000 | 2.741e-08 |
| 1 | python_callback | 0.000/0.000 | 0.000/0.000 | 0.002/0.000 | 0.003/0.000 | 0.048/0.000 | 0.046/0.000 | 0.002/0.000 | 0.015/0.000 | 0.001/0.000 | 0.001/0.000 | 46 | 1 | 44 | 7 | 0.000/0.000 | 2.741e-08 |
| 8 | expr_legacy | 0.005/0.000 | 0.000/0.000 | 0.010/0.001 | 0.018/0.001 | 0.061/0.001 | 0.053/0.000 | 0.007/0.001 | 0.016/0.000 | 0.001/0.000 | 0.003/0.000 | 46 | 1 | 44 | 8 | 0.000/0.000 | 3.124e-08 |
| 8 | atom_native | 0.007/0.002 | 0.001/0.000 | 0.015/0.003 | 0.209/0.011 | 0.080/0.005 | 0.062/0.003 | 0.016/0.002 | 0.017/0.001 | 0.002/0.000 | 0.004/0.000 | 46 | 1 | 44 | 8 | 0.001/0.000 | 3.124e-08 |
| 8 | python_callback | 0.000/0.000 | 0.000/0.000 | 0.002/0.000 | 0.006/0.000 | 0.069/0.000 | 0.067/0.000 | 0.002/0.000 | 0.029/0.000 | 0.003/0.000 | 0.003/0.000 | 46 | 1 | 44 | 8 | 0.000/0.000 | 3.124e-08 |
| 32 | expr_legacy | 0.018/0.001 | 0.000/0.000 | 0.032/0.002 | 0.092/0.004 | 0.094/0.003 | 0.085/0.002 | 0.009/0.001 | 0.022/0.000 | 0.002/0.000 | 0.012/0.000 | 46 | 1 | 44 | 7 | 0.001/0.000 | 4.808e-09 |
| 32 | atom_native | 0.018/0.001 | 0.001/0.000 | 0.040/0.004 | 0.373/0.044 | 0.105/0.004 | 0.091/0.004 | 0.014/0.001 | 0.022/0.000 | 0.002/0.000 | 0.013/0.001 | 46 | 1 | 44 | 7 | 0.004/0.002 | 4.808e-09 |
| 32 | python_callback | 0.000/0.000 | 0.000/0.000 | 0.003/0.000 | 0.033/0.001 | 0.134/0.001 | 0.131/0.001 | 0.003/0.000 | 0.070/0.001 | 0.028/0.000 | 0.012/0.000 | 46 | 1 | 44 | 7 | 0.001/0.000 | 4.808e-09 |
| 128 | expr_legacy | 0.063/0.001 | 0.002/0.000 | 0.122/0.004 | 1.118/0.027 | 0.788/0.003 | 0.776/0.003 | 0.013/0.000 | 0.054/0.000 | 0.060/0.001 | 0.550/0.002 | 56 | 1 | 54 | 9 | 0.001/0.000 | 4.933e-08 |
| 128 | atom_native | 0.065/0.001 | 0.002/0.000 | 0.119/0.002 | 0.802/0.082 | 0.834/0.034 | 0.818/0.033 | 0.017/0.001 | 0.063/0.002 | 0.005/0.001 | 0.584/0.034 | 56 | 1 | 54 | 9 | 0.002/0.000 | 4.933e-08 |
| 128 | python_callback | 0.000/0.000 | 0.001/0.000 | 0.010/0.000 | 0.449/0.005 | 1.013/0.012 | 1.008/0.012 | 0.006/0.000 | 0.308/0.001 | 0.426/0.007 | 0.536/0.006 | 56 | 1 | 54 | 9 | 0.001/0.000 | 4.933e-08 |
