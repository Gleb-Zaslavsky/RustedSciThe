param(
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
if ($AllowDirtySource -and $Action -ne "Record") {
    throw "-AllowDirtySource is valid only with -Action Record."
}

# Keep this focused gate independent from the full BVP_sci and all-solver suites.
$runner = Join-Path $PSScriptRoot "perf_gates_run.ps1"
$runArgs = @{
    Action = $Action
    Machine = $Machine
    Group = "bvp-sci-parameter-loop"
    Warmups = $Warmups
    Runs = $Runs
    MaxRelativeMad = $MaxRelativeMad
    Tolerance = $Tolerance
}
if ($Force) { $runArgs.Force = $true }
if ($AllowDirtySource) { $runArgs.AllowDirtySource = $true }
if (-not [string]::IsNullOrWhiteSpace($Toolchain)) { $runArgs.Toolchain = $Toolchain }

& $runner @runArgs
