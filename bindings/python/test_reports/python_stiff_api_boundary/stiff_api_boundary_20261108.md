# Python StiffAPI Boundary Benchmark

Workload: 32-state diffusion chain, dense BDF / ExprLegacy, `[0, 0.1]`, 11 samples.
Provenance: Python=3.14.6, rustedscithe=0.1.5, package_path=`D:\RUST\RustProjects_\RustedSciThe_experimental4\bindings\python\python\rustedscithe\__init__.py`, native_extension=`D:\RUST\RustProjects_\RustedSciThe_experimental4\bindings\python\python\rustedscithe\_native.pyd`
All values are median/MAD milliseconds. Request construction includes Python/native input validation; prepare includes facade dispatch and symbolic/native preparation. `solve_wall_ms` is end-to-end Python wall time. Native stage telemetry is nested and must not be added to wall time. Trajectory copy measures `trajectory_arrays()` NumPy conversion; result extraction measures detached final-state and telemetry access.

| route | request_build_validate_ms | prepare_dispatch_ms | solve_wall_ms | native_residual_ms | native_jacobian_ms | native_linear_ms | trajectory_numpy_copy_ms | result_extract_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BDF / ExprLegacy / Dense | 0.0384/0.0025 | 0.8404/0.0576 | 4.1063/0.1639 | 0.0858/0.0037 | 0.0108/0.0003 | 0.9028/0.0066 | 0.0191/0.0006 | 0.0559/0.0019 |
