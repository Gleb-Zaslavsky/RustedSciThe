param(
    [Parameter(Mandatory)][ValidateSet("Record", "Compare")][string]$Action,
    [Parameter(Mandatory)][string]$Machine,
    [ValidateSet("bdf", "radau", "lsode2", "be")]
    [string[]]$Solvers = @("bdf", "radau", "lsode2", "be"),
    [ValidateRange(0, 1000)][int]$Warmups = 20,
    [ValidateRange(3, 10001)][int]$Runs = 301,
    [ValidateRange(0.0, 1.0)][double]$MaxRelativeMad = 0.05,
    [ValidateRange(0.0, 1.0)][double]$Tolerance = 0.10,
    [switch]$Force,
    [switch]$AllowDirtySource,
    [string]$Toolchain,
    [switch]$PlanOnly
)

# Addressed first-tier IVP performance gate. It intentionally reuses the
# calibrated per-solver groups and never expands to perf-gates --all.
$ErrorActionPreference = "Stop"
if ($Force -and $Action -ne "Record") { throw "-Force is valid only with -Action Record." }
if ($AllowDirtySource -and $Action -ne "Record") {
    throw "-AllowDirtySource is valid only with -Action Record."
}

$runner = Join-Path $PSScriptRoot "perf_gates_first_tier.ps1"
if ($PlanOnly) {
    Write-Host "IVP perf-gates plan: action=$Action machine=$Machine warmups=$Warmups runs=$Runs"
    foreach ($solver in $Solvers) { Write-Host " - $solver" }
    exit 0
}

$results = [System.Collections.Generic.List[object]]::new()
foreach ($solver in $Solvers) {
    Write-Host "`n=== IVP perf-gates: $solver / $Action ==="
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $status = "passed"
    $message = ""
    try {
        $arguments = @{
            Solver = $solver
            Action = $Action
            Machine = $Machine
            Warmups = $Warmups
            Runs = $Runs
            MaxRelativeMad = $MaxRelativeMad
            Tolerance = $Tolerance
        }
        if ($Force) { $arguments.Force = $true }
        if ($AllowDirtySource) { $arguments.AllowDirtySource = $true }
        if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
            $arguments.Toolchain = $Toolchain
        }
        & $runner @arguments
        if ($LASTEXITCODE -ne 0) { throw "$solver exited with code $LASTEXITCODE" }
    }
    catch {
        $status = "failed"
        $message = $_.Exception.Message
    }
    $watch.Stop()
    $results.Add([pscustomobject][ordered]@{
        solver = $solver
        action = $Action.ToLowerInvariant()
        status = $status
        duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        detail = $message
    })
}

Write-Host "`nIVP performance-gate summary"
$results | Format-Table -AutoSize -Wrap
$failed = @($results | Where-Object { $_.status -eq "failed" }).Count
if ($failed -gt 0) {
    Write-Error "$failed IVP performance-gate solver group(s) failed."
    exit 1
}
Write-Host "IVP PERF-GATES COMPLETE: $($Action.ToUpperInvariant()) passed for $($Solvers.Count) solver(s)."
exit 0
