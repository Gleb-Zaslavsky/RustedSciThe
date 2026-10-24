param(
    [string]$PythonPath,
    [ValidateRange(1, 64)][int]$CargoBuildJobs = 8,
    [switch]$SkipSymbolicInfrastructure,
    [switch]$SkipFacadeTests,
    [switch]$PlanOnly
)

# Correctness-only release gate for the complete maintained IVP test surface.
# It runs library tests only (never `cargo bench` or examples), includes ignored
# parity/stress stories, and keeps stdout/stderr attached to the terminal.
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-ScipyPython([string]$Requested) {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($Requested)) { $candidates.Add($Requested) }
    if (-not [string]::IsNullOrWhiteSpace($env:IVP_SCIPY_PYTHON)) {
        $candidates.Add($env:IVP_SCIPY_PYTHON)
    }
    $candidates.Add((Join-Path $repoRoot ".venv-gavin-win\Scripts\python.exe"))
    $candidates.Add((Join-Path $repoRoot ".venv\Scripts\python.exe"))
    $candidates.Add((Join-Path $repoRoot ".venv/bin/python"))
    $candidates.Add("python3")
    $candidates.Add("python")

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
        $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue
        if ($null -ne $command) { return $command.Source }
    }
    throw "SciPy parity requires Python with NumPy and SciPy. Pass -PythonPath or set IVP_SCIPY_PYTHON."
}

$steps = [System.Collections.Generic.List[object]]::new()
if (-not $SkipSymbolicInfrastructure) {
    $steps.Add([pscustomobject]@{ name = "symbolic_ivp"; filter = "symbolic::symbolic_ivp" })
    $steps.Add([pscustomobject]@{ name = "symbolic_ivp_aot"; filter = "symbolic::symbolic_ivp_aot" })
    $steps.Add([pscustomobject]@{ name = "symbolic_ivp_generated"; filter = "symbolic::symbolic_ivp_generated" })
    $steps.Add([pscustomobject]@{ name = "symbolic_ivp_telemetry"; filter = "symbolic::ivp_telemetry" })
    $steps.Add([pscustomobject]@{ name = "symbolic_codegen_parameterized_ivp"; filter = "symbolic::codegen::tests::codegen_parameterized_ivp_tests" })
    $steps.Add([pscustomobject]@{ name = "symbolic_codegen_ivp_performance"; filter = "symbolic::codegen::tests::codegen_ivp_performance_tests" })
    $steps.Add([pscustomobject]@{ name = "symbolic_codegen_ivp_backend_parity"; filter = "symbolic::codegen::tests::codegen_ivp_backend_comparison_tests" })
}
$steps.Add([pscustomobject]@{ name = "ivp_workload_fixtures"; filter = "numerical::ivp_workloads::" })
$steps.Add([pscustomobject]@{ name = "lsode2"; filter = "numerical::LSODE2::" })
$steps.Add([pscustomobject]@{ name = "bdf"; filter = "numerical::BDF::" })
$steps.Add([pscustomobject]@{ name = "radau"; filter = "numerical::Radau::" })
$steps.Add([pscustomobject]@{ name = "backward_euler"; filter = "numerical::BE::" })
if (-not $SkipFacadeTests) {
    $steps.Add([pscustomobject]@{ name = "ode_api2"; filter = "numerical::ODE_api2::" })
    $steps.Add([pscustomobject]@{ name = "nonstiff_facade"; filter = "numerical::NonStiff_api::" })
    $steps.Add([pscustomobject]@{ name = "rosenbrock"; filter = "numerical::Rosenbrock::" })
}

if ($PlanOnly) {
    Write-Host "Complete IVP release test plan (all selected tests include ignored; no benches/examples):"
    $steps | Format-Table name, filter -AutoSize
    exit 0
}

$env:IVP_SCIPY_PYTHON = Resolve-ScipyPython $PythonPath
Write-Host "Validating SciPy parity Python: $env:IVP_SCIPY_PYTHON"
& $env:IVP_SCIPY_PYTHON -c "import sys, numpy, scipy; print(sys.executable); print('NumPy', numpy.__version__); print('SciPy', scipy.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "The configured Python cannot import NumPy and SciPy; IVP tests were not started."
}

$results = [System.Collections.Generic.List[object]]::new()
Push-Location $repoRoot
try {
    foreach ($step in $steps) {
        Write-Host "`n=== IVP release tests: $($step.name) ==="
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        $cargoArgs = @(
            "test", "--release", "--lib", "--no-default-features",
            "--jobs", $CargoBuildJobs, $step.filter, "--",
            "--include-ignored", "--nocapture", "--test-threads=1"
        )
        & cargo @cargoArgs
        $exitCode = [int]$LASTEXITCODE
        $watch.Stop()
        $results.Add([pscustomobject][ordered]@{
            step = $step.name
            status = $(if ($exitCode -eq 0) { "passed" } else { "failed" })
            exit_code = $exitCode
            duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        })
    }
}
finally {
    Pop-Location
}

Write-Host "`nComplete IVP release test summary"
$results | Format-Table -AutoSize
$failed = @($results | Where-Object { $_.exit_code -ne 0 }).Count
if ($failed -gt 0) {
    Write-Error "$failed IVP release test step(s) failed."
    exit 1
}
Write-Host "ALL IVP RELEASE TESTS COMPLETE: PASS for $($results.Count) test group(s)."
exit 0
