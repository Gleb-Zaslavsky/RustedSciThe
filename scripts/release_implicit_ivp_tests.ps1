param(
    [string]$PythonPath,
    [ValidateRange(1, 64)][int]$CargoBuildJobs = 8,
    [switch]$PlanOnly
)

# Release correctness run for the four implicit IVP solver families. This is
# deliberately a cargo test-only script: no examples, benches, or perf-gates.
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

    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($candidate in $candidates) {
        $executable = $null
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $executable = (Resolve-Path -LiteralPath $candidate).Path
        }
        else {
            $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue
            if ($null -ne $command) { $executable = $command.Source }
        }

        if ($null -eq $executable -or -not $seen.Add($executable)) { continue }
        & $executable -c "import sys, numpy, scipy; print(sys.executable); print('NumPy', numpy.__version__); print('SciPy', scipy.__version__)" | Out-Host
        if ($LASTEXITCODE -eq 0) { return $executable }
        Write-Warning "Python candidate cannot import NumPy and SciPy: $executable"
    }

    throw "SciPy parity tests require Python with NumPy and SciPy. Pass -PythonPath or set IVP_SCIPY_PYTHON."
}

$steps = @(
    [pscustomobject]@{ name = "LSODE2"; filter = "numerical::LSODE2::" },
    [pscustomobject]@{ name = "Backward Euler"; filter = "numerical::BE::" },
    [pscustomobject]@{ name = "Radau"; filter = "numerical::Radau::" },
    [pscustomobject]@{ name = "BDF"; filter = "numerical::BDF::" }
)

if ($PlanOnly) {
    Write-Host "Implicit IVP Release tests (ignored included; tests only):"
    $steps | Format-Table name, filter -AutoSize
    exit 0
}

$env:IVP_SCIPY_PYTHON = Resolve-ScipyPython $PythonPath
Write-Host "SciPy parity interpreter: $env:IVP_SCIPY_PYTHON"

$results = [System.Collections.Generic.List[object]]::new()
Push-Location $repoRoot
try {
    foreach ($step in $steps) {
        Write-Host "`n=== Release tests: $($step.name) (including ignored) ==="
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
            solver = $step.name
            status = $(if ($exitCode -eq 0) { "passed" } else { "failed" })
            exit_code = $exitCode
            duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        })
    }
}
finally {
    Pop-Location
}

Write-Host "`nImplicit IVP Release test summary"
$results | Format-Table -AutoSize
$failed = @($results | Where-Object { $_.exit_code -ne 0 }).Count
if ($failed -gt 0) {
    Write-Error "$failed implicit IVP test group(s) failed."
    exit 1
}

Write-Host "IMPLICIT IVP RELEASE TESTS COMPLETE: PASS for $($results.Count) solver groups."
exit 0
