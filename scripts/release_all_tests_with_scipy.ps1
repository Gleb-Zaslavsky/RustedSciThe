param(
    [string]$PythonPath,
    [ValidateRange(1, 256)]
    [int]$TestThreads = 1
)

# Run only the crate library tests in release mode, including ignored tests.
# Library stdout/stderr stay attached to the console for live diagnostics.
# Limit parallel compilation so the run remains usable on a shared workstation.
$ErrorActionPreference = "Stop"
$cargoBuildJobs = 8

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = $env:IVP_SCIPY_PYTHON
}
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $repoRoot ".venv-gavin-win\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "SciPy parity tests need Python with NumPy and SciPy. Set IVP_SCIPY_PYTHON or pass -PythonPath. Tried: $PythonPath"
}

$env:IVP_SCIPY_PYTHON = (Resolve-Path -LiteralPath $PythonPath).Path
$env:SOMELINALG_PYTHON = $env:IVP_SCIPY_PYTHON
Write-Host "Validating SciPy parity Python: $env:IVP_SCIPY_PYTHON"
& $env:IVP_SCIPY_PYTHON -c "import sys, numpy, scipy; print(sys.executable); print('NumPy', numpy.__version__); print('SciPy', scipy.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "The configured Python could not import NumPy and SciPy; Rust tests were not started."
}

Write-Host "Running release library tests, including ignored tests; Cargo build jobs: $cargoBuildJobs; test threads: $TestThreads."
cargo test --release --lib --jobs $cargoBuildJobs -- --include-ignored --test-threads $TestThreads
$testExitCode = $LASTEXITCODE
if ($testExitCode -ne 0) {
    Write-Error "Release test suite failed with exit code $testExitCode."
}
exit $testExitCode
