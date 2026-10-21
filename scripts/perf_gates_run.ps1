param(
    [Parameter(Mandatory)][ValidateSet("Record", "Compare")][string]$Action,
    [Parameter(Mandatory)][string]$Machine,
    [string]$Solver,
    [string]$Group,
    [string]$Scenario,
    [int]$Warmups = 10,
    [int]$Runs = 101,
    [double]$MaxRelativeMad = 0.05,
    [double]$Tolerance = 0.10,
    [switch]$Force,
    [switch]$AllowDirtySource,
    [string]$Toolchain
)

$ErrorActionPreference = "Stop"
$selectorCount = 0
if (-not [string]::IsNullOrWhiteSpace($Solver)) { $selectorCount++ }
if (-not [string]::IsNullOrWhiteSpace($Group)) { $selectorCount++ }
if (-not [string]::IsNullOrWhiteSpace($Scenario)) { $selectorCount++ }
if ($selectorCount -ne 1) {
    throw "Specify exactly one of -Solver, -Group or -Scenario."
}
if ($Warmups -lt 0 -or $Runs -lt 3) {
    throw "Warmups must be nonnegative and Runs must be at least 3."
}
if ($Force -and $Action -ne "Record") {
    throw "-Force is valid only with -Action Record."
}
if ($AllowDirtySource -and $Action -ne "Record") {
    throw "-AllowDirtySource is valid only with -Action Record."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$previousToolchain = $env:RUSTUP_TOOLCHAIN

try {
    if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
        $env:RUSTUP_TOOLCHAIN = $Toolchain
    }

    $cargoArgs = @("run", "--release", "--features", "perf-gates", "--bin", "perf-gates", "--", $Action.ToLowerInvariant())
    if (-not [string]::IsNullOrWhiteSpace($Solver)) { $cargoArgs += @("--solver", $Solver) }
    if (-not [string]::IsNullOrWhiteSpace($Group)) { $cargoArgs += @("--group", $Group) }
    if (-not [string]::IsNullOrWhiteSpace($Scenario)) { $cargoArgs += @("--scenario", $Scenario) }
    $cargoArgs += @("--machine", $Machine, "--warmups", $Warmups, "--runs", $Runs, "--max-relative-mad", $MaxRelativeMad)
    if ($Action -eq "Record") {
        $cargoArgs += @("--tolerance", $Tolerance)
        if ($Force) { $cargoArgs += "--force" }
        if ($AllowDirtySource) { $cargoArgs += "--allow-dirty-source" }
    }

    Push-Location $repoRoot
    try {
        Write-Host "perf-gates $($Action.ToLowerInvariant()): machine=$Machine warmups=$Warmups runs=$Runs max-relative-mad=$MaxRelativeMad"
        & cargo @cargoArgs
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            throw "perf-gates $($Action.ToLowerInvariant()) failed with exit code $code. Inspect the run report printed above."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $previousToolchain) {
        Remove-Item Env:RUSTUP_TOOLCHAIN -ErrorAction SilentlyContinue
    }
    else {
        $env:RUSTUP_TOOLCHAIN = $previousToolchain
    }
}
