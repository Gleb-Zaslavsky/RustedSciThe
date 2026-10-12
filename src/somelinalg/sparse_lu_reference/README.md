# Sparse LU Reference Corpus

This repository-only corpus isolates the sparse direct solve used by the
SciPy BVP path from nonlinear iteration, mesh refinement, and symbolic setup.
The Python reference is SciPy 1.18.1 `scipy.sparse.linalg.splu` (SuperLU).

Fixtures are square, nonsymmetric, diagonally dominant CSC matrices with dense
state blocks on the mesh diagonal, neighboring block couplings, and a small
global border for parameters/boundary conditions. Rust exports exact triplets,
RHS, and known solution consumed by Python. Cross-language checks include CSC
structural count, residual, and solution error as well as stage timings.

Run the fast Rust correctness gate:

```powershell
cargo test --lib --no-default-features somelinalg::sparse_lu_reference -- --nocapture
cargo test --lib --no-default-features somelinalg::bvp_sparse_lu_tests -- --nocapture
```

Run the Python input-schema and BVP compatibility tests:

```powershell
& .\.venv-gavin-win\Scripts\python.exe -m unittest discover -s src\somelinalg\sparse_lu_reference\tests -v
```

The cross-language ignored story uses the configured Python executable and
checks the pinned SciPy version:

```powershell
$env:SOMELINALG_PYTHON = (Resolve-Path .\.venv-gavin-win\Scripts\python.exe).Path
cargo test --lib --no-default-features bvp_shaped_fixture_matches_scipy_splu_reference -- --ignored --nocapture
```

The opt-in SuperLU control can additionally compare the complete solution
vector and materialized factor fill against SciPy on the same matrix/RHS. This
is a debug correctness gate, not a performance run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_superlu_bridge.ps1
$env:SUPERLU_LIBRARY = (Resolve-Path 'test_reports\somelinalg_sparse_lu_reference\native\superlu_bridge.dll').Path
$env:SOMELINALG_PYTHON = (Resolve-Path '.venv-gavin-win\Scripts\python.exe').Path
$env:SOMELINALG_SPARSE_LU_CAPTURE_FIXTURE = (Resolve-Path 'test_reports\BVP_sci_sparse_lu_diagnostics_20261012\fixtures\captured_bvp_jacobian_1022.txt').Path
cargo test --lib --no-default-features --features superlu somelinalg::sparse_lu_reference_tests::superlu_bridge_matches_scipy_factorization_and_solution_on_shared_matrix -- --ignored --nocapture --test-threads=1
```

The debug report is written under
`test_reports/somelinalg_sparse_lu_reference/debug/`. SciPy OpenBLAS is forced
to one thread for this gate; the SuperLU bridge uses its sequential bundled
CBLAS. Vendor/version and thread policy are printed, so the different BLAS
implementations remain explicit.

Run the Rust low-level microbenchmark and export fixtures for SciPy:

```powershell
$env:SOMELINALG_SPARSE_LU_EXPORT_DIR = "test_reports/somelinalg_sparse_lu_reference/fixtures"
cargo bench --no-default-features --bench somelinalg_sparse_lu_reference
```

Run Python against each exported `*.txt` fixture:

```powershell
Get-ChildItem $env:SOMELINALG_SPARSE_LU_EXPORT_DIR -Filter *.txt | ForEach-Object {
  & $env:SOMELINALG_PYTHON src\somelinalg\sparse_lu_reference\splu_benchmark.py $_.FullName
}
```

`splu` performs symbolic and numeric factorization together through its public
API. Its cold factorization plus solve is the direct cross-language comparison.
The Rust reusable kernel additionally reports symbolic setup and warm numeric
refactor separately. That warm-refactor number has no direct `splu` equivalent
and must not be presented as apples-to-apples parity.

The experimental Rust kernel is connected to BVP_sci only through the
explicit opt-in `BvpSciSparseFactorization::ReusableFaerExperimental` policy;
the default remains `StandardFaer`. The experimental adapter retains CSC
matrix values between refactors, so its memory cost must be measured. Shared
`somelinalg::banded` implementations are unchanged.

The separately built SuperLU backend can be selected by BVP_sci with Cargo
feature `superlu`, `BvpSciSparseFactorization::SuperLu`, or builder
`.with_superlu()`. Set `SUPERLU_LIBRARY` to the bridge DLL/shared object. This
is opt-in only; `BvpSciSolver::new` and StandardFaer remain unchanged. Its
persistent C factor handle supports repeated Newton RHS solves without
refactorization. See `src/numerical/BVP_sci/SUPERLU_BACKEND.md` for platform
build instructions and current validation limits.

The older BVP_sci linear benchmark entry point remains as a compatibility
shim; its legacy fixture header and output keys are handled by the same runner,
so the SuperLU timing implementation is maintained only once.

## Optional Native SuperLU Control

An opt-in SuperLU 6.0.1 control is available behind non-default Cargo feature
`superlu`. It is exposed through the checked `somelinalg::superlu` facade and
uses a small private C ABI bridge built for the current platform. The build
script pins the upstream source commit and uses SuperLU's bundled CBLAS; SciPy's
wheel uses OpenBLAS, so these timing differences are diagnostic rather than a
controlled BLAS comparison.

In PowerShell, build the bridge and run its correctness gate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_superlu_bridge.ps1
$env:SUPERLU_LIBRARY = (Resolve-Path 'test_reports\somelinalg_sparse_lu_reference\native\superlu_bridge.dll').Path
cargo test --lib --no-default-features --features superlu somelinalg::sparse_lu_superlu_experimental::tests::superlu_control_solves_the_same_sparse_fixture -- --ignored --nocapture
```

Run the release same-matrix microbenchmark against an optional captured BVP
Jacobian:

```powershell
$env:SOMELINALG_SPARSE_LU_CAPTURE_FIXTURE = (Resolve-Path 'test_reports\BVP_sci_sparse_lu_diagnostics_20261012\fixtures\captured_bvp_jacobian_1022.txt').Path
$env:SOMELINALG_SCIPY_BLAS_THREADS = '1'
$env:SOMELINALG_SPARSE_LU_REPEATS = '9'
cargo bench --no-default-features --features superlu --bench somelinalg_sparse_lu_reference -- --noplot
```

For multiple captured Jacobians, use a semicolon-separated list in
`SOMELINALG_SPARSE_LU_CAPTURE_FIXTURES`; the singular variable above remains
supported for compatibility. The benchmark runs each capture through both
faer strategies and the SuperLU control. It records backend/BLAS provenance;
do not combine timing rows across distinct BLAS builds as if they were a
single controlled run.

The bridge reports native storage counts separately from CSC `L.nnz/U.nnz`.
The latter reproduce SciPy's materialization semantics and are directly
comparable. The 2026-10-12 captured-matrix result and its BLAS caveat are
recorded in `TODO.md`; no production backend change follows from that single
measurement.

## Initial Release Slice

The first 9-sample release slice passed the Python fixture suite (`3/3`) and
the cross-language correctness story. Its synthetic same-matrix results are
archived in `TODO.md`: faer cold factor+solve was faster than SciPy at
dimensions 50 and 514, while SciPy was faster at 1,538. The reusable kernel's
cold path was slower than ordinary faer at 514/1,538, but subsequent
same-pattern numeric refactors were substantially cheaper. That is a candidate
for repeated BVP Newton factorizations, not evidence to switch the production
backend: real captured BVP Jacobians and amortized same-pattern workloads are
still required. The cold-total versus separately measured symbolic/warm stages
also needs better attribution before treating the lifecycle timings as
additive.

The first BVP-level debug A/B passed on a parameterized linear problem,
including a restart on a larger mesh. The release full-solve continuation A/B
also passed (5 repeats, 8 rebinds): standard faer and reusable faer produced
identical final mesh/state and the same counters (16 numeric factorizations,
8 symbolic analyses, 8 symbolic reuses). The reusable route measured 1.612 ms
versus 1.648 ms for the solve series, while its numeric-factorization scope was
0.029 ms versus 0.048 ms. This is a small end-to-end gain on one workload, not
a basis for changing the default. Captured production Jacobians and retained
memory remain unmeasured. See the compact report linked in `TODO.md`.
