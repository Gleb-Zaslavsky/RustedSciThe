# Python Boundary Benchmarks

These scripts measure the installed Python facade boundary, not the Rust
solver core in isolation. Use a clean environment with `rustedscithe`, NumPy
and the declared benchmark dependencies installed. To benchmark a release
wheel, invoke the script with that wheel's virtual-environment Python while
working outside the checkout import path.

## Boundary And NumPy Microbenchmark

`python_boundary_overhead.py` measures a minimal constant-expression PyO3
round trip and NumPy/Python sequence materialization for increasing vector
sizes. The PyO3 row includes native constant evaluation and argument/result
conversion; it is a lower-bound boundary proxy, not a claim to isolate pure
dispatch instructions. It is useful alongside solver benchmarks, where
construction, callbacks, preparation and numerical work are measured in their
real API context.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_overhead.py `
  --sizes 1,8,64,256,1024 --repeats 9 --batch 2000 `
  --output $env:TEMP\python_boundary_overhead.md
```

## Matched BVP_sci Boundary Matrix

`python_boundary_matrix.py` uses one parameterized exact problem for all
routes: `y' = rate`, `y(0) = 0`, `y(1) = 1`, with exact solution `y=x` and
`rate=1`. It compares ExprLegacy/Lambdify and AtomNative/Lambdify, adds a
Numerical route using Python residual callbacks, and can optionally include
ExprLegacy/AtomNative AOT builds.

From PowerShell with an installed wheel environment:

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_matrix.py `
  --nodes 32,64,128 --repeats 5 --continuation 8 `
  --output $env:TEMP\python_boundary_matrix.md
```

Add `--include-aot` only when `tcc` or `gcc` is available on `PATH`. AOT rows
use `RebuildAlways` with fresh artifact directories and include cold build/link
timings. On Windows,
the loaded DLL may prevent immediate deletion of generated temporary artifacts;
cleanup is best-effort. If a directory remains, the process has exited and the
temporary directory can then be removed manually.

The report separates Python expression construction and a standalone NumPy
`tolist()` materialization sample from solver construction, first-solve wall
time, NumPy result conversion and warm
parameter-rebind series. Native residual/Jacobian/factorization timings and
their call/factorization counts are reported alongside wall time; nested stage
timings are not additive to it. The Numerical callback row includes Python/GIL
crossings and is not a matched symbolic-frontend speed comparison. Use repeated
release baselines for performance claims; this script is diagnostic evidence.

`python_boundary_bvp.py` remains a smaller one-problem diagnostic that compares
symbolic Lambdify against a Python callback route.

## Compact Faithful Banded Boundary

`python_banded.py` measures `somelinalg`'s compact faithful banded API without
constructing a dense coefficient matrix. It reports packed assembly,
factorization, warm single-RHS solve, multiple-RHS solve and explicit
iterative-refinement stages separately, together with retained storage bytes.
The result is Python/NumPy boundary evidence, not a replacement for a Rust
core benchmark.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_banded.py `
  --dimensions 32,128,512 --bandwidths 1,2 --repeats 5 `
  --output $env:TEMP\python_banded.md
```

## IVP Boundary and Continuation

`python_boundary_ivp.py` compares ExprLegacy and AtomNative on the same dense
BDF problem `y'=-rate*y`, and adds a separate Python callback route. It times
expression creation, NumPy-to-native input
materialization, construction, explicit preparation, first solve, NumPy result
conversion, and prepared parameter-continuation calls. Every segment is checked
against the piecewise exact solution. The BDF continuation API combines
parameter rebinding and integration in one call, so the report labels that
combined boundary cost rather than inventing a rebind-only estimate. Both total
continuation wall time and nested Rust solve/residual stages are reported. Native
stage timings and discrete solver counters are reported separately from Python
wall time. The callback row includes Python/GIL crossings within the residual
and Jacobian work; it does not claim to isolate pure dispatch overhead. Its
continuation uses `restart()` with a changed Python closure, unlike the
symbolic route's native parameter-rebind operation.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_ivp.py `
  --repeats 7 --continuation 8 --segment-end 0.25 `
  --output $env:TEMP\python_boundary_ivp.md
```

`python_boundary_ivp_scale.py` complements that scalar continuation workload
with dimension-scaled exact BDF problems (`n=1,8,32,128` by default). It
compares ExprLegacy and AtomNative on the same diagonal decay systems and
reports an analytical Python callback route separately. The callback row
crosses the GIL during residual/Jacobian evaluation; it is not a frontend
parity claim. Each route checks the exact exponential endpoint and requires
native residual/Jacobian/linear-solve/factorization counters plus stage timings.
It also reports `wall - Rust solve` as a signed timing residual. This is not
pure Python CPU time: it includes Python dispatch, PyO3 conversions and native
orchestration omitted from solver timers, so it must not be added to Rust
stages or interpreted as an independently measured Python-only duration.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_ivp_scale.py `
  --dimensions 1,8,32,128 --repeats 5 --end-time 0.2 `
  --output $env:TEMP\python_boundary_ivp_scale.md
```

`python_boundary_radau.py` runs the same exact scalar IVP through Radau's
symbolic ExprLegacy/AtomNative routes and its separate Python callback route.
Its repeated symbolic calls use Radau's prepared parameter-rebind API with an
explicit `y0`; this is a parameter sweep over the configured interval, not a
time-advancing continuation series. The callback row changes a Python closure
and calls `continue_with_initial_state`, so it shares the mathematical fixture
but has a distinct lifecycle. The report includes native residual, Jacobian,
factorization and linear stages/counters alongside Python wall times, with
exact-solution gates on the initial solve and every repeated result.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_radau.py `
  --repeats 7 --sweep 8 --segment-end 0.25 `
  --output $env:TEMP\python_boundary_radau.md
```

`python_boundary_lsode2.py` varies the native LSODE2 controller (`bdf`,
`adams`, `auto`) and ExprLegacy/AtomNative on the same exact scalar decay
problem. It measures explicit preparation, one initial solve, NumPy trajectory
materialization and time-advancing prepared parameter continuation. LSODE2
timing snapshots are cumulative, so continuation stage timing is computed from
adjacent snapshots; discrete counters shown in the table are for the first
solve. Automatic mode reports which family actually executed. The `evaluation_*`
counters are explicitly tagged by `evaluation_scope`; do not substitute the
bridge-level `linear_solve_calls` key for native faithful inner-loop solves.
Older wheels without scope-aware keys are reported as
`not_exposed_by_wheel` / `n/a`, not zero. A measured zero remains distinct
from an unavailable key.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_lsode2.py `
  --controllers bdf,adams,auto --repeats 7 --continuation 8 `
  --segment-end 0.2 --output $env:TEMP\python_boundary_lsode2.md
```

## Nonlinear and Fitting Boundary

`python_boundary_nonlinear_fitting.py` measures Newton root solving and
prepared symbolic least-squares fitting with both symbolic frontends. It also
measures Gavin's sampled-observation fitting interface. Each workload has an
independent exact synthetic-data gate. Newton's explicit `prepare()` is
separated from solving; `FittingSolver` prepares in its constructor, and Gavin
combines builder setup and preparation, so their preparation scopes are
intentionally identified rather than treated as equal. Fitting continuation
reuses the prepared equation graph. Gavin is a different sampled-data contract
and is descriptive, not a direct speed competitor to equation least squares.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_nonlinear_fitting.py `
  --repeats 7 --samples 64 --warm-solves 8 --continuation 8 `
  --output $env:TEMP\python_boundary_nonlinear_fitting.md
```

`python_boundary_nonlinear_aot.py` adds a matched medium/large nonlinear-system
matrix outside BVP_sci. For each dimension and symbolic frontend it solves the
same independent square system `x_i^2 - p_i = 0` through Lambdify and cold AOT,
and checks every initial and warm rebind result against the exact positive root.
Every AOT repeat uses `RebuildAlways` and a fresh artifact directory, so its
preparation time includes lowering, compile and link rather than a warm artifact
cache hit. The report includes the warm series, average warm call and complete
constructor+preparation+solve lifecycle beside Lambdify, making amortization
visible. It reports an estimated warm-rebind break-even from median preparation
overhead and median warm-call savings; `not reached` is explicit when AOT warm
calls are not faster. This estimate is diagnostic, not a hard statistical
threshold. Rust solve stages are nested in Python wall time. This is
wrapper-facing lifecycle evidence, not a core compiler microbenchmark; use
native core benches for backend-only comparisons. Initial-solve and warm-series
`wall - Rust` residuals are reported separately, with the same attribution
caveat as the IVP benchmark.

```powershell
& $wheelPython .\bindings\python\benchmarks\python_boundary_nonlinear_aot.py `
  --dimensions 4,16,32 --repeats 3 --warm-solves 8 `
  --output $env:TEMP\python_boundary_nonlinear_aot.md
```

These additions extend wrapper-level evidence only. They do not replace native
Rust release benchmarks, and no universal frontend or solver ranking should be
inferred from a single fixture. Use the same wheel, interpreter, machine and
environment when comparing saved reports.

The boundary harnesses treat required counters as required data: if a detailed
telemetry snapshot omits one, the run fails with the route and key instead of
printing a fabricated zero. Optional timing stages are rendered as `n/a` when
the route does not expose them. Cumulative LSODE2 timing deltas are differenced
between snapshots and must be monotonic; missing stages remain unavailable.

The scaled BDF and nonlinear/AOT reports include a `Runtime:` provenance line
with Python/package versions and resolved package/native-extension paths. For
release claims, verify both paths point into the clean wheel environment's
`site-packages`; a checkout/editable path is diagnostic-only. Results and their
interpretation are recorded in
[`PYTHON_BOUNDARY_STORY_TESTS.md`](../PYTHON_BOUNDARY_STORY_TESTS.md).
