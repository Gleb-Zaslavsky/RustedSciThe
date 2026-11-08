# Python StiffAPI Boundary Benchmark

Workload: 32-state diffusion chain, dense BDF / ExprLegacy, `[0, 0.1]`, 11 samples.
Provenance: Python=3.14.6, rustedscithe=0.1.6, package_path=`C:\Users\user\AppData\Local\Temp\rustedscithe-wheel-contract-dcad0e03e3f24207b5e6258a795cba00\Lib\site-packages\rustedscithe\__init__.py`, native_extension=`C:\Users\user\AppData\Local\Temp\rustedscithe-wheel-contract-dcad0e03e3f24207b5e6258a795cba00\Lib\site-packages\rustedscithe\_native.pyd`
All values are median/MAD milliseconds. Request construction includes Python/native input validation; prepare includes facade dispatch and symbolic/native preparation. `solve_wall_ms` is end-to-end Python wall time. Native stage telemetry is nested and must not be added to wall time. Trajectory copy measures `trajectory_arrays()` NumPy conversion; result extraction measures detached final-state and telemetry access.

| route | request_build_validate_ms | prepare_dispatch_ms | solve_wall_ms | native_residual_ms | native_jacobian_ms | native_linear_ms | trajectory_numpy_copy_ms | result_extract_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BDF / ExprLegacy / Dense | 0.0114/0.0022 | 0.2747/0.0677 | 0.1195/0.0103 | 0.0244/0.0011 | 0.0012/0.0001 | 0.0161/0.0005 | 0.0028/0.0004 | 0.0153/0.0018 |
