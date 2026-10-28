param(
    [Parameter(Mandatory)][ValidateSet("Record", "Compare", "RecordCompare")][string]$Action,
    [Parameter(Mandatory)][string]$Machine,
    [ValidateRange(0, 1000)][int]$Warmups = 10,
    [ValidateRange(3, 10001)][int]$Runs = 101,
    [ValidateRange(0.0, 1.0)][double]$MaxRelativeMad = 0.05,
    [ValidateRange(0.0, 1.0)][double]$Tolerance = 0.10,
    [switch]$Force,
    [switch]$AllowDirtySource,
    [string]$Toolchain
)

$ErrorActionPreference = "Stop"
if ($Action -eq "Compare" -and ($Force -or $AllowDirtySource)) {
    throw "-Force and -AllowDirtySource apply only to actions containing Record."
}

$runner = Join-Path $PSScriptRoot "perf_gates_run.ps1"
$runArgs = @{
    Action = $Action
    Machine = $Machine
    Group = "ivp-parameter-loop"
    Warmups = $Warmups
    Runs = $Runs
    MaxRelativeMad = $MaxRelativeMad
    Tolerance = $Tolerance
}
if ($Force) { $runArgs.Force = $true }
if ($AllowDirtySource) { $runArgs.AllowDirtySource = $true }
if (-not [string]::IsNullOrWhiteSpace($Toolchain)) { $runArgs.Toolchain = $Toolchain }

& $runner @runArgs
