param(
    [Parameter(Mandatory)][ValidateSet(
        "bdf", "radau", "lsode2", "be", "bvp-sci", "bvp-damp",
        "nonlinear-systems", "optimisation"
    )][string]$Solver,
    [Parameter(Mandatory)][ValidateSet("Record", "Compare")][string]$Action,
    [Parameter(Mandatory)][string]$Machine,
    [int]$Warmups = 10,
    [int]$Runs = 101,
    [double]$MaxRelativeMad = 0.05,
    [double]$Tolerance = 0.10,
    [switch]$Force,
    [switch]$AllowDirtySource,
    [string]$Toolchain
)

$ErrorActionPreference = "Stop"
if ($Force -and $Action -ne "Record") { throw "-Force is valid only with -Action Record." }
if ($AllowDirtySource -and $Action -ne "Record") { throw "-AllowDirtySource is valid only with -Action Record." }

$groups = @{
    "bdf" = @("bdf-matrix")
    "radau" = @("radau-matrix")
    "lsode2" = @("lsode2-matrix")
    "be" = @("be-matrix", "large-system")
    "bvp-sci" = @("bvp-sci-large")
    "bvp-damp" = @("bvp-damp-large")
    "nonlinear-systems" = @("nonlinear-large")
    "optimisation" = @("optimization-large")
}

$runner = Join-Path $PSScriptRoot "perf_gates_run.ps1"
$failures = [System.Collections.Generic.List[string]]::new()
foreach ($group in $groups[$Solver]) {
    Write-Host "`n=== $Solver / group $group / $Action ==="
    $runArgs = @{
        Action = $Action
        Machine = $Machine
        Group = $group
        Warmups = $Warmups
        Runs = $Runs
        MaxRelativeMad = $MaxRelativeMad
        Tolerance = $Tolerance
    }
    if ($Force) { $runArgs.Force = $true }
    if ($AllowDirtySource) { $runArgs.AllowDirtySource = $true }
    if (-not [string]::IsNullOrWhiteSpace($Toolchain)) { $runArgs.Toolchain = $Toolchain }

    try {
        & $runner @runArgs
    }
    catch {
        $failures.Add("${group}: $($_.Exception.Message)")
    }
}

if ($failures.Count -gt 0) {
    $failureList = $failures -join "`n - "
    throw "Some $Solver perf-gates groups did not complete:`n - $failureList"
}
Write-Host "`n$Solver $($Action.ToLowerInvariant()) completed for: $($groups[$Solver] -join ', ')"
