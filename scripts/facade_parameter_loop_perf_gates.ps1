param(
    [Parameter(Mandatory)][ValidateSet("Record", "Compare", "RecordCompare")][string]$Action,
    [Parameter(Mandatory)][string]$Machine,
    [ValidateSet("All", "BvpApi", "StiffApi")][string]$Facade = "All",
    [ValidateRange(0, 1000)][int]$Warmups = 10,
    [ValidateRange(3, 10001)][int]$Runs = 101,
    [ValidateRange(0.0, 1.0)][double]$MaxRelativeMad = 0.05,
    [ValidateRange(0.0, 1.0)][double]$Tolerance = 0.10,
    [switch]$Force,
    [switch]$AllowDirtySource,
    [string]$Toolchain,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
if ($Action -eq "Compare" -and ($Force -or $AllowDirtySource)) {
    throw "-Force and -AllowDirtySource apply only to actions containing Record."
}

$groups = switch ($Facade) {
    "BvpApi" { @("bvp-api-parameter-loop") }
    "StiffApi" { @("stiff-api-parameter-loop") }
    default { @("bvp-api-parameter-loop", "stiff-api-parameter-loop") }
}
$actions = if ($Action -eq "RecordCompare") { @("Record", "Compare") } else { @($Action) }
if ($PlanOnly) {
    Write-Host "Facade parameter-loop perf-gates plan: facade=$Facade machine=$Machine"
    foreach ($group in $groups) {
        foreach ($step in $actions) {
            Write-Host " - $step group=$group (warmups=$Warmups runs=$Runs)"
        }
    }
    exit 0
}

$runner = Join-Path $PSScriptRoot "perf_gates_run.ps1"
foreach ($group in $groups) {
    foreach ($step in $actions) {
        $runArgs = @{
            Action = $step
            Machine = $Machine
            Group = $group
            Warmups = $Warmups
            Runs = $Runs
            MaxRelativeMad = $MaxRelativeMad
            Tolerance = $Tolerance
        }
        if ($step -eq "Record") {
            if ($Force) { $runArgs.Force = $true }
            if ($AllowDirtySource) { $runArgs.AllowDirtySource = $true }
        }
        if (-not [string]::IsNullOrWhiteSpace($Toolchain)) { $runArgs.Toolchain = $Toolchain }

        Write-Host "`n=== Facade parameter-loop perf-gates: facade=$Facade group=$group action=$step ==="
        & $runner @runArgs
    }
}
