# Python Boundary Story Tests

This document records correctness-gated Python-boundary workloads and their
interpretation. These are not Rust-core performance benchmarks. Their purpose
is to measure the Python facade/PyO3/NumPy boundary around real native calls;
Rust solver timings and counters are included only as attribution and
correctness context. The normative scope is defined in
[`PYTHON_ARCHITECTURAL_GUIDELINE.md`](PYTHON_ARCHITECTURAL_GUIDELINE.md#5-measure-python-boundary-cost).

## Reporting Contract

- Run each workload against a clean environment containing the exact installed
  wheel, with the working directory outside the checkout.
- Preserve the benchmark's exact-solution or residual gate and verify native
  counters so timing rows represent the same numerical work.
- Report Python-observed wall time separately from native telemetry. A signed
  `wall - Rust` residual is an attribution estimate, not pure Python CPU time;
  it may contain PyO3 conversion, uninstrumented native orchestration and
  measurement noise.
- Record interpreter/package versions and resolved Python/native-extension
  paths in each report. A checkout/editable path is not release-wheel evidence.
- Use Rust-core story tests and benches for algorithmic fidelity, solver
  rankings, backend comparisons and numerical-core performance claims.

## 2026-10-17: Scaled BDF IVP And Nonlinear Cold AOT

Source: user-provided console report for
`python_boundary_ivp_scale.py` and `python_boundary_nonlinear_aot.py`.
The report was generated before runtime provenance and explicit wall-minus-
native residual columns were added. The transcript invokes `$wheelPython` but
does not show its resolved path, package version or extension path. Therefore
the measurements are useful diagnostic evidence, but **release-wheel
provenance is unverified** and they must not be labeled as a clean-wheel
baseline.

### Scaled BDF Boundary Workload

Matched diagonal system: `y_i' = -rate_i y_i`, `y_i(0)=1`, `t_end=0.2`,
dimensions `1, 8, 32, 128`, five repeats. All routes passed the exact
exponential endpoint gate; maximum reported error was below `4.94e-8`.
Within each dimension, ExprLegacy, AtomNative and Python-callback routes had
matching discrete solver counts. At `n=128`, all routes reported 56 residual
calls, one Jacobian call, 54 linear solves and 9 LU factorizations; smaller
dimensions used 46 residual calls, one Jacobian call and 44 linear solves.

| n | route | Python solve wall median/MAD ms | Rust solve median/MAD ms | residual calls | linear solves | LU | max exact error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ExprLegacy | 0.331/0.008 | 0.275/0.004 | 46 | 44 | 7 | 2.741e-8 |
| 1 | AtomNative | 0.340/0.025 | 0.273/0.018 | 46 | 44 | 7 | 2.741e-8 |
| 1 | Python callback | 0.298/0.012 | 0.273/0.008 | 46 | 44 | 7 | 2.741e-8 |
| 8 | ExprLegacy | 0.700/0.002 | 0.647/0.001 | 46 | 44 | 8 | 3.124e-8 |
| 8 | AtomNative | 0.781/0.025 | 0.704/0.014 | 46 | 44 | 8 | 3.124e-8 |
| 8 | Python callback | 0.691/0.005 | 0.672/0.003 | 46 | 44 | 8 | 3.124e-8 |
| 32 | ExprLegacy | 3.527/0.006 | 3.468/0.008 | 46 | 44 | 7 | 4.808e-9 |
| 32 | AtomNative | 3.768/0.045 | 3.686/0.044 | 46 | 44 | 7 | 4.808e-9 |
| 32 | Python callback | 3.620/0.009 | 3.595/0.008 | 46 | 44 | 7 | 4.808e-9 |
| 128 | ExprLegacy | 102.023/0.747 | 101.905/0.761 | 56 | 54 | 9 | 4.933e-8 |
| 128 | AtomNative | 101.640/0.234 | 101.511/0.272 | 56 | 54 | 9 | 4.933e-8 |
| 128 | Python callback | 101.829/0.119 | 101.771/0.122 | 56 | 54 | 9 | 4.933e-8 |

The Python-callback row is not a symbolic-frontend comparison: its numerical
callbacks cross the GIL. The native factorization stage dominates the `n=128`
solve, so these measurements primarily show that Python wall time is close to
the native solve time for this fixture. AtomNative preparation was higher than
ExprLegacy in these samples (for example, `5.572 ms` versus `4.193 ms` at
`n=128`); this small, diagonal workload does not establish a general frontend
ranking and should not override Rust-core evidence on large/coupled symbolic
systems.

### Nonlinear Cold-AOT Amortization Workload

Matched dense system: `x_i^2 - p_i = 0`, exact positive root, dimensions
`4, 16, 32`, three repeats and eight warm parameter rebinds. Lambdify and AOT
rows matched in residual/Jacobian/linear-solve counts and exact-root output.
Each AOT sample used a fresh artifact directory and `RebuildAlways`; reported
preparation includes lowering, compilation and linking.

| n | frontend | Lambdify warm call ms | AOT warm call ms | cold AOT prepare ms | estimated break-even rebinds | max root error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | ExprLegacy | 0.052/0.007 | 0.052/0.004 | 216.695/5.605 | not reached | 1.130e-13 |
| 4 | AtomNative | 0.042/0.001 | 0.040/0.000 | 222.133/0.053 | 139782 | 1.130e-13 |
| 16 | ExprLegacy | 0.205/0.000 | 0.233/0.026 | 221.759/0.959 | not reached | 2.220e-16 |
| 16 | AtomNative | 0.194/0.001 | 0.184/0.005 | 224.154/1.393 | 23593 | 2.220e-16 |
| 32 | ExprLegacy | 0.808/0.004 | 0.800/0.004 | 228.735/7.747 | 28239 | 3.864e-14 |
| 32 | AtomNative | 0.863/0.006 | 0.814/0.016 | 230.174/1.441 | 4620 | 3.864e-14 |

These estimates are fixture-specific and based on only three repeats. Eight
warm calls do not amortize cold compilation. This benchmark measures Python
API lifecycle and attribution, not whether AOT is generally faster; use the
Rust-core AOT/Lambdify evidence for that conclusion.

## 2026-10-17: Local 0.1.4 Release Wheel

Built the Windows ABI3 wheel from the current checkout with maturin's release
profile, installed it into a new temporary venv, and ran the benchmarks from
`%TEMP%` so the checkout was not the import root. This is a **local release
wheel measurement**, not a published-PyPI measurement. Both report provenance
lines resolve `rustedscithe` and `_native.pyd` under the clean venv's
`site-packages`. `pip check` reported no broken requirements.

- Wheel: `rustedscithe-0.1.4-cp310-abi3-win_amd64.whl`
- SHA-256: `CC90972D65DDD6CAB0895708ACBF208D0AC4BF2B3CBFDEA563B766D24655F4F6`
- Runtime: CPython `3.14.6`, NumPy `2.5.4`
- Environment: `C:\Users\user\AppData\Local\Temp\rustedscithe-014-boundary-clean`
- Full reports: `test_reports/python_boundary_0.1.4/ivp_scale_release_wheel.md`
  and `test_reports/python_boundary_0.1.4/nonlinear_aot_release_wheel.md`

### BDF Boundary Result

The exact exponential endpoint gate passed for all routes and sizes (maximum
error `4.933e-8`). Counts matched between ExprLegacy, AtomNative and Python
callbacks at each dimension: `n=1/8/32` used 46 residual calls, one Jacobian,
44 linear solves; `n=128` used 56, one, and 54 respectively. LU counts were
7/8/7/9. At `n=128`, solve wall time was `0.788 ms` ExprLegacy,
`0.834 ms` AtomNative and `1.013 ms` Python callback, while Rust solve timing
was `0.776/0.818/1.008 ms`. The corresponding median `wall - Rust` residuals
were `0.013/0.017/0.006 ms`; the callback route additionally performs its
residual/Jacobian work through Python and holds the GIL. These are observed
boundary costs for this diagonal BDF fixture, not a solver or frontend speed
ranking.

| n | route | prepare wall ms | solve wall ms | Rust solve ms | wall - Rust ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | ExprLegacy | 0.007/0.000 | 0.053/0.001 | 0.046/0.001 | 0.007/0.000 |
| 1 | AtomNative | 0.011/0.000 | 0.051/0.000 | 0.043/0.000 | 0.008/0.000 |
| 1 | Python callback | 0.003/0.000 | 0.048/0.000 | 0.046/0.000 | 0.002/0.000 |
| 8 | ExprLegacy | 0.018/0.001 | 0.061/0.001 | 0.053/0.000 | 0.007/0.001 |
| 8 | AtomNative | 0.209/0.011 | 0.080/0.005 | 0.062/0.003 | 0.016/0.002 |
| 8 | Python callback | 0.006/0.000 | 0.069/0.000 | 0.067/0.000 | 0.002/0.000 |
| 32 | ExprLegacy | 0.092/0.004 | 0.094/0.003 | 0.085/0.002 | 0.009/0.001 |
| 32 | AtomNative | 0.373/0.044 | 0.105/0.004 | 0.091/0.004 | 0.014/0.001 |
| 32 | Python callback | 0.033/0.001 | 0.134/0.001 | 0.131/0.001 | 0.003/0.000 |
| 128 | ExprLegacy | 1.118/0.027 | 0.788/0.003 | 0.776/0.003 | 0.013/0.000 |
| 128 | AtomNative | 0.802/0.082 | 0.834/0.034 | 0.818/0.033 | 0.017/0.001 |
| 128 | Python callback | 0.449/0.005 | 1.013/0.012 | 1.008/0.012 | 0.006/0.000 |

The AtomNative preparation time is not monotonic relative to ExprLegacy across
these repeated runs (`n=8/32` AtomNative is higher, while `n=128` is lower).
Do not infer a general symbolic-preparation conclusion from this simple
diagonal fixture; the core frontend benchmarks remain authoritative.

### Nonlinear Cold-AOT Result

Five repeats, dimensions `4/16/32`, eight warm rebinds, both frontends. All
routes passed the exact-root gate and paired Lambdify/AOT counters matched.
Cold AOT preparation medians ranged from `221.315` to `241.017 ms`, including
fresh lowering, compilation and link. With eight warm rebinds, no pair showed
a stable amortization; only AtomNative `n=32` had a lower AOT warm-call median
(`0.024 ms` versus `0.027 ms`), yielding a highly fixture-specific estimate
of about `105,498` rebinds to recover cold preparation. All other rows reported
`not reached` because their AOT warm call was not faster. This measures the
Python-visible lifecycle and does not replace Rust-core AOT performance
evidence.

| n | frontend | Lambdify warm call ms | AOT warm call ms | AOT prepare ms | estimated break-even rebinds |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | ExprLegacy | 0.011/0.002 | 0.017/0.004 | 241.017/3.258 | not reached |
| 4 | AtomNative | 0.007/0.000 | 0.009/0.000 | 221.315/1.382 | not reached |
| 16 | ExprLegacy | 0.012/0.000 | 0.020/0.001 | 225.563/1.203 | not reached |
| 16 | AtomNative | 0.013/0.000 | 0.014/0.000 | 235.113/4.148 | not reached |
| 32 | ExprLegacy | 0.024/0.001 | 0.040/0.010 | 231.371/2.160 | not reached |
| 32 | AtomNative | 0.027/0.000 | 0.024/0.000 | 225.617/5.310 | 105498 |

### Reproduction

The wheel and venv were created with these PowerShell steps (no upload):

```powershell
$wheelOut = Join-Path $env:TEMP 'rustedscithe-014-boundary-wheel'
$wheelEnv = Join-Path $env:TEMP 'rustedscithe-014-boundary-clean'
$basePython = (Resolve-Path '.\.venv-gavin-win\Scripts\python.exe').Path
& $basePython -m maturin build --manifest-path .\bindings\python\Cargo.toml --release --out $wheelOut
$wheel = (Get-ChildItem $wheelOut -Filter 'rustedscithe-0.1.4-*.whl' | Select-Object -First 1).FullName
& $basePython -m venv $wheelEnv
$wheelPython = Join-Path $wheelEnv 'Scripts\python.exe'
& $wheelPython -m pip install $wheel
$benchDir = (Resolve-Path '.\bindings\python\benchmarks').Path
Push-Location $env:TEMP
try {
  & $wheelPython (Join-Path $benchDir 'python_boundary_ivp_scale.py') --dimensions 1,8,32,128 --repeats 9 --end-time 0.2 --output (Join-Path $env:TEMP 'python_boundary_ivp_scale_release.md')
  if ($LASTEXITCODE -ne 0) { throw 'BDF boundary story failed' }
  & $wheelPython (Join-Path $benchDir 'python_boundary_nonlinear_aot.py') --dimensions 4,16,32 --repeats 5 --warm-solves 8 --output (Join-Path $env:TEMP 'python_boundary_nonlinear_aot_release.md')
  if ($LASTEXITCODE -ne 0) { throw 'Nonlinear AOT boundary story failed' }
} finally {
  Pop-Location
}
```

The `Runtime:` line in both reports is a required provenance check: package and
native-extension paths must resolve beneath `$wheelEnv\Lib\site-packages`.
